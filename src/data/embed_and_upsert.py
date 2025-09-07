import argparse, json, os, torch
import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pinecone import Pinecone

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = torch.sum(masked, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def load_model():
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tok, model

def encode_batch(texts, tok, model, device):
    enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
        pooled = mean_pool(outputs.last_hidden_state, enc["attention_mask"])
        vecs = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return vecs.cpu().numpy()

def upsert(pc: Pinecone, index_name: str, ids, vectors, metadatas):
    index = pc.Index(index_name)
    # Pinecone v5 client expects list of dicts
    items = [{"id": str(i), "values": v.tolist(), "metadata": m} for i, v, m in zip(ids, vectors, metadatas)]
    index.upsert(items)

def main(input_path: str):
    load_dotenv()
    index_name = os.getenv("PINECONE_INDEX")
    if not index_name:
        raise RuntimeError("PINECONE_INDEX not set in .env")
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set in .env")
    pc = Pinecone(api_key=api_key)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_model()
    model.to(device)

    ids, texts, metas = [], [], []
    with open(input_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)
            ids.append(ex.get("id") or str(len(ids)))
            texts.append(ex["text"])
            metas.append({"question": ex.get("question",""), "answer": ex.get("answer",""), "topic": ex.get("topic",""), "source": ex.get("source","")})

    B = 32
    all_vecs = []
    for i in tqdm(range(0, len(texts), B), desc="Embedding"):
        batch = texts[i:i+B]
        vecs = encode_batch(batch, tok, model, device)
        all_vecs.append(vecs)
    vecs = np.vstack(all_vecs) if all_vecs else np.zeros((0,768))

    upsert(pc, index_name, ids, vecs, metas)
    print(f"Upserted {len(ids)} vectors to Pinecone index '{index_name}'.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL file from prepare_medquad")
    args = ap.parse_args()
    main(args.input)

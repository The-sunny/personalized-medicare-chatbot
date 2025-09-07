import argparse, json, pickle
from rank_bm25 import BM25Okapi

def build_bm25(input_path: str, output_path: str):
    corpus = []
    ids = []
    with open(input_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)
            ids.append(ex.get("id"))
            text = (ex.get("question","") + " " + ex.get("answer","")).strip()
            corpus.append(text.split())
    bm25 = BM25Okapi(corpus)
    with open(output_path, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids}, f)
    print(f"Saved BM25 index with {len(ids)} docs to {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    build_bm25(args.input, args.output)

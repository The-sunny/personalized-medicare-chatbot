import argparse, json, pandas as pd, pathlib

def to_clean(in_path: str, out_path: str):
    rows = []
    with open(in_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            doc_id = ex.get("id") or ex.get("_id")
            q = ex.get("question") or ex.get("title") or ""
            a = ex.get("answer") or ex.get("content") or ""
            topic = ex.get("topic") or ex.get("category") or "general"
            source = ex.get("source") or "unknown"
            text = (q + " " + a).strip()
            rows.append({
                "id": doc_id,
                "question": q,
                "answer": a,
                "topic": topic,
                "source": source,
                "text": text
            })
    df = pd.DataFrame(rows).dropna(subset=["text"]).reset_index(drop=True)
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_path, orient="records", lines=True)
    print(f"Wrote {len(df)} records to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    to_clean(args.input, args.output)

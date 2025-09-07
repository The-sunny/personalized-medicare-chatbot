import argparse, json, random

def simulate_uncertainty(answer: str) -> float:
    # toy heuristic: shorter answers => lower confidence
    L = max(1, len(answer.split()))
    base = min(1.0, 0.2 + L/50.0)
    noise = random.uniform(-0.1, 0.1)
    return max(0.0, min(1.0, base + noise))

def run(input_path: str, out_flags: str, threshold: float = 0.55):
    flagged = 0
    with open(input_path, "r") as f, open(out_flags, "w") as out:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)
            conf = simulate_uncertainty(ex.get("answer",""))
            if conf < threshold:
                ex["active_learning_flag"] = True
                ex["model_confidence"] = conf
                out.write(json.dumps(ex)+"\n")
                flagged += 1
    print(f"Flagged {flagged} examples for review (< {threshold:.2f} confidence). Wrote to {out_flags}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--flags", required=True)
    ap.add_argument("--threshold", type=float, default=0.55)
    args = ap.parse_args()
    run(args.input, args.flags, args.threshold)

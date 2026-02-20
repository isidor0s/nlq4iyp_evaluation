"""
Model Comparison Judge — Evaluate LLM answers vs IYP ground truth using Groq.

Scoring: 0-2 per dimension (Completeness, Hallucination, Specificity).
Max total = 6 per question.

Usage:
    python data/model_comparison/model_judge.py            # full run
"""
import os, json, time, glob, argparse
from dotenv import load_dotenv
load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SYSTEM_PROMPT = """You are an expert Internet infrastructure evaluator.
You compare a MODEL_ANSWER against GROUND_TRUTH from the IYP (Internet Yellow Pages) knowledge graph.

Score each dimension on a 0-2 scale:

1) COMPLETENESS — Does the answer cover the key entities/values in the ground truth?
   2 = all or nearly all ground truth values present
   1 = some values present, but key ones missing
   0 = none of the ground truth values present, or answer is a refusal

2) HALLUCINATION — Does the model invent facts that contradict or are absent from the ground truth?
   2 = no fabricated data, answer aligns with ground truth
   1 = minor inaccuracies or unverifiable claims mixed in
   0 = heavily fabricated values or confidently wrong data

3) SPECIFICITY — Does the answer give concrete values (ASNs, IPs, counts, server names) vs vague text?
   2 = provides specific values matching or close to ground truth
   1 = partially specific, mixes concrete data with vague statements
   0 = entirely vague or generic, no concrete data points

Respond with ONLY a JSON object (no markdown, no extra text):
{"completeness": 0, "hallucination": 0, "specificity": 0, "total": 0, "explanation": "..."}"""

SCORE_KEYS = ["completeness", "hallucination", "specificity"]

def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["questions"]

def parse_answers(path, n):
    """Parse answer file by splitting on Q-headers (Q1:, Q2:, etc.) instead of ---,
    because model answers often contain --- as markdown separators."""
    import re
    with open(path, encoding="utf-8") as f:
        content = f.read()
    # Find all Q-header positions: "Q1:", "Q2:", etc. at start of line
    pattern = re.compile(r'^Q(\d+):', re.MULTILINE)
    matches = list(pattern.finditer(content))
    answers = []
    for idx, m in enumerate(matches):
        # Text starts after the Q-header line
        header_end = content.index('\n', m.start()) + 1 if '\n' in content[m.start():] else len(content)
        # Text ends at the next Q-header or end of file
        text_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        text = content[header_end:text_end].strip()
        # Remove trailing --- separators between questions
        text = re.sub(r'\n---\s*$', '', text).strip()
        answers.append(text if text else "[No answer]")
    while len(answers) < n:
        answers.append("[No answer]")
    return answers[:n]

def discover_models(d):
    return {os.path.splitext(os.path.basename(p))[0]: p
            for p in sorted(glob.glob(os.path.join(d, "*.txt")))
            if not os.path.basename(p).startswith("_")}

def judge(client, question, gt_summary, gt_raw, model_answer):
    prompt = (f"Question: \"{question}\"\n\n"
              f"GROUND_TRUTH (from IYP):\n{gt_summary}\n"
              f"Raw: {json.dumps(gt_raw, default=str)}\n\n"
              f"MODEL_ANSWER:\n{model_answer}")
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile", temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}])
        r = json.loads(resp.choices[0].message.content)
        vals = [r.get(k, 0) for k in SCORE_KEYS]
        r["total"] = sum(vals)
        return r
    except Exception as e:
        return {k: 0 for k in SCORE_KEYS} | {"total": 0, "explanation": f"Error: {e}"}

def print_table(results, questions):
    models = sorted(results)
    hdr = f"{'Model':<22}" + "".join(f"  {q['id']:^8}" for q in questions) + f"  {'TOTAL':>6}"
    print(f"\n{'='*len(hdr)}\nScores per question (sum of 3 dimensions, max 6)\n{hdr}\n{'-'*len(hdr)}")
    for m in models:
        scores = [r["total"] for r in results[m]]
        row = f"{m:<22}" + "".join(f"  {s:^8}" for s in scores)
        row += f"  {sum(scores)/len(scores):>6.1f}"
        print(row)
    print("=" * len(hdr))

    # Per-dimension breakdown
    print(f"\nPer-dimension averages (0-2 scale):")
    hdr2 = f"{'Model':<22}  {'COMPL':>6}  {'HALLU':>6}  {'SPEC':>6}  {'AVG':>6}"
    print(hdr2 + "\n" + "-" * len(hdr2))
    for m in models:
        avgs = {k: sum(r.get(k,0) for r in results[m]) / len(results[m]) for k in SCORE_KEYS}
        overall = sum(avgs.values()) / len(avgs)
        print(f"{m:<22}  {avgs['completeness']:>6.1f}  {avgs['hallucination']:>6.1f}"
              f"  {avgs['specificity']:>6.1f}  {overall:>6.1f}")

def main():
    ap = argparse.ArgumentParser(description="Model judge (Groq)")
    ap.add_argument("--questions", default=os.path.join(SCRIPT_DIR, "questions.json"))
    ap.add_argument("--answers-dir", default=os.path.join(SCRIPT_DIR, "answers"))
    ap.add_argument("--output", default=os.path.join(SCRIPT_DIR, "results.json"))
    args = ap.parse_args()

    questions = load_questions(args.questions)
    models = discover_models(args.answers_dir)
    if not models:
        print(f"No .txt files in {args.answers_dir}/"); return
    all_answers = {m: parse_answers(p, len(questions)) for m, p in models.items()}

    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    results = {}
    for m, ans in all_answers.items():
        print(f"\nEvaluating {m}...")
        results[m] = []
        for i, q in enumerate(questions):
            r = judge(client, q["question"], q["ground_truth_summary"], q["ground_truth"], ans[i])
            print(f"  {q['id']}: C={r.get('completeness',0)} H={r.get('hallucination',0)} "
                  f"S={r.get('specificity',0)} → {r['total']}/6")
            results[m].append(r)
            time.sleep(0.5)

    print_table(results, questions)
    output = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "scoring": "0-2 per dimension, 3 dimensions, max 6 per question",
        "dimensions": SCORE_KEYS,
        "questions": [{"id": q["id"], "task_id": q["task_id"],
                       "difficulty": q["difficulty"], "question": q["question"]}
                      for q in questions],
        "results": results,
        "summary": {
            m: {k: round(sum(r.get(k,0) for r in rs)/len(rs), 2) for k in SCORE_KEYS}
            for m, rs in results.items()
        }
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[SAVED] {args.output}")

if __name__ == "__main__":
    main()

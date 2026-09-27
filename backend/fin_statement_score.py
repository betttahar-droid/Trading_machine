"""
GPU side of backend/fin_statement_lab.py: a small instruction model reads an anonymised two-year balance sheet and
income statement and answers A (EPS higher next year) or B (lower); p_up = P(A) from the first token's
log-probabilities, renormalised over A/B.

    python fin_statement_score.py input.jsonl Qwen/Qwen2.5-3B-Instruct-AWQ qwen3b
"""

import csv
import json
import math
import sys

SYSTEM = "You are an experienced financial analyst."
QUESTION = ("Below are the financial statements of an anonymous company for two consecutive fiscal years "
            "(USD millions, except per-share data).\n\n{table}\n\n"
            "Based only on these statements, will the company's earnings per share be higher or lower next year "
            "than in the current year?\nA) higher\nB) lower\nAnswer with a single letter.")


def main(path: str, model: str, name: str):
    from vllm import LLM, SamplingParams
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    llm = LLM(model=model, max_model_len=2048, gpu_memory_utilization=0.90, max_logprobs=20)
    tok = llm.get_tokenizer()
    ids = {L: {tok.encode(v, add_special_tokens=False)[0] for v in (L, " " + L)} for L in "AB"}
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": QUESTION.format(table=r["table"])}],
                                       tokenize=False, add_generation_prompt=True) for r in rows]
    outs = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20))
    with open(f"fs_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "p_up", "coverage"])
        for r, o in zip(rows, outs):
            lp = o.outputs[0].logprobs[0]
            pa = sum(math.exp(lp[i].logprob) for i in ids["A"] if i in lp)
            pb = sum(math.exp(lp[i].logprob) for i in ids["B"] if i in lp)
            w.writerow([r["id"], round(pa / (pa + pb), 4) if pa + pb > 0 else 0.5, round(pa + pb, 4)])


if __name__ == "__main__":
    main(*sys.argv[1:4])

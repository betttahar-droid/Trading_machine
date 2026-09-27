"""
GPU side of backend/biotech_lab.py: a small instruction model reads the start of each press release and answers two
multiple-choice questions; answers are read from the first token's log-probabilities (no text generated).

  outcome   A positive (primary endpoint met / clearly positive efficacy), B mixed or unclear,
            C negative (endpoint missed, trial failed or stopped, serious safety problem), D no new trial results
  phase     A Phase 1, B Phase 2, C Phase 3 or pivotal, D not stated
  evidence  A strong (significant primary endpoint, clear effect), B moderate, C weak or spun (trends, no p-values,
            secondary endpoints, subgroups, small open-label), D no efficacy results

    python biotech_score.py docs.jsonl Qwen/Qwen2.5-3B-Instruct-AWQ qwen3b
"""

import csv
import json
import math
import sys

SYSTEM = "You are a pharmacist and biotech analyst. You read company press releases about clinical trials."
OUTCOME = ("Start of a press release filed with the SEC by {name} on {date}:\n\n\"{text}\"\n\n"
           "Does it report new clinical trial results, and how did the trial turn out for the company's drug?\n"
           "A) positive: the primary endpoint was met, or clearly positive efficacy data\n"
           "B) mixed or unclear results\n"
           "C) negative: the primary endpoint was not met, the trial failed or was stopped, or a serious safety problem\n"
           "D) it does not report new clinical trial results\n"
           "Answer with a single letter.")
PHASE = ("Start of a press release filed with the SEC by {name} on {date}:\n\n\"{text}\"\n\n"
         "Which phase is the clinical trial whose results are reported?\n"
         "A) Phase 1\nB) Phase 2\nC) Phase 3 or pivotal\nD) not stated or no trial results\n"
         "Answer with a single letter.")
EVIDENCE = ("Start of a press release filed with the SEC by {name} on {date}:\n\n\"{text}\"\n\n"
            "As a critical pharmacist, how strong is the clinical evidence it reports (ignore the company's tone)?\n"
            "A) strong: the primary endpoint was met with statistical significance and a clear effect size\n"
            "B) moderate: significant but small effect, or only some endpoints met\n"
            "C) weak or spun: no p-values, only trends, numerical or clinically meaningful improvements, secondary "
            "endpoints, subgroups, or a small open-label study\n"
            "D) no efficacy results reported\n"
            "Answer with a single letter.")
LETTERS = "ABCD"


def main(docs_path: str, model: str, name: str, chars: int = 2500):
    from vllm import LLM, SamplingParams
    docs = [json.loads(line) for line in open(docs_path, encoding="utf-8")]
    llm = LLM(model=model, max_model_len=2048, gpu_memory_utilization=0.90, max_logprobs=20)
    tok = llm.get_tokenizer()
    ids = {L: {tok.encode(v, add_special_tokens=False)[0] for v in (L, " " + L)} for L in LETTERS}

    def prompts(tmpl):
        out = []
        for d in docs:
            text = d["text"][:chars]
            msg = [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": tmpl.format(name=d["name"], date=d["file_date"], text=text)}]
            p = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            while len(tok.encode(p)) > 1900 and len(text) > 500:          # keep inside the context window
                text = text[:int(len(text) * 0.8)]
                msg[1]["content"] = tmpl.format(name=d["name"], date=d["file_date"], text=text)
                p = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            out.append(p)
        return out

    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    res = {}
    for q, tmpl in (("out", OUTCOME), ("phase", PHASE), ("ev", EVIDENCE)):
        outs = llm.generate(prompts(tmpl), sp)
        res[q] = []
        for o in outs:
            lp = o.outputs[0].logprobs[0]
            pr = [sum(math.exp(lp[i].logprob) for i in ids[L] if i in lp) for L in LETTERS]
            tot = sum(pr) or 1.0
            res[q].append([x / tot for x in pr])
    with open(f"bio_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["adsh"] + [f"{q}_{L}" for q in ("out", "phase", "ev") for L in LETTERS])
        for d, a, b, c in zip(docs, res["out"], res["phase"], res["ev"]):
            w.writerow([d["adsh"]] + [round(x, 4) for x in a + b + c])


if __name__ == "__main__":
    main(*sys.argv[1:4])

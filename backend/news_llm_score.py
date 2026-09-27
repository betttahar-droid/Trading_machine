"""
GPU side of backend/news_llm_lab.py: score news posts with an LLM through vLLM.

For each post the model is asked which way the news moves Bitcoin over the next few hours and must answer with one
letter, A (strongly down) .. G (strongly up). The answer is read from the first token's log-probabilities, so no text
is generated: score = sum over letters of p(letter) x value (A = -3 .. G = +3, probabilities renormalised over the
seven letters), impact = 1 - p(D, "no effect"), coverage = total probability on the seven letters.

    python news_llm_score.py posts.jsonl Qwen/Qwen2.5-72B-Instruct-AWQ qwen72b
"""

import csv
import json
import math
import sys
import time

SYSTEM = "You are an experienced crypto trader."
PROMPT = ("Breaking news posted on Telegram at {when} UTC:\n\n\"{text}\"\n\n"
          "How will this news move the price of Bitcoin over the next few hours?\n"
          "A) strongly down\nB) down\nC) slightly down\nD) no effect\nE) slightly up\nF) up\nG) strongly up\n"
          "Answer with a single letter.")
LETTERS = "ABCDEFG"


def main(posts_path: str, model: str, name: str):
    from vllm import LLM, SamplingParams

    posts = [json.loads(line) for line in open(posts_path, encoding="utf-8")]
    llm = LLM(model=model, max_model_len=2048, gpu_memory_utilization=0.90, max_logprobs=20)
    tok = llm.get_tokenizer()
    ids = {L: {tok.encode(v, add_special_tokens=False)[0] for v in (L, " " + L)} for L in LETTERS}
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": PROMPT.format(when=time.strftime("%Y-%m-%d %H:%M", time.gmtime(p["ts"])),
                                                   text=p["text"])}],
        tokenize=False, add_generation_prompt=True) for p in posts]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20))
    print(f"{name}: {len(prompts)} posts in {time.time() - t0:.0f} s", flush=True)
    with open(f"scores_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "ts", "score", "impact", "coverage"] + [f"p_{L}" for L in LETTERS])
        for p, o in zip(posts, outs):
            lp = o.outputs[0].logprobs[0]
            probs = [sum(math.exp(lp[i].logprob) for i in ids[L] if i in lp) for L in LETTERS]
            cov = sum(probs)
            q = [x / cov for x in probs] if cov > 0 else [0, 0, 0, 1, 0, 0, 0]
            score = sum(qk * (k - 3) for k, qk in enumerate(q))
            w.writerow([p["id"], p["ts"], round(score, 4), round(1 - q[3], 4), round(cov, 4)] + [round(x, 4) for x in q])


if __name__ == "__main__":
    main(*sys.argv[1:4])

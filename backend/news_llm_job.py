"""
Unattended GPU job for backend/news_llm_lab.py, started by a rented instance's onstart script (no SSH needed):

  1. downloads the models in the background while it waits for the posts
  2. receives posts.jsonl.gz over HTTP (POST /<JOB_TOKEN>/posts) or, after 4 minutes without it, scrapes
     @WatcherGuru itself (backend/telegram_data.py) and filters it like news_llm_lab.prepare_posts
  3. scores the posts with each model (backend/news_llm_score.py, one process per model so GPU memory is freed)
  4. publishes status and results over HTTP (GET /<JOB_TOKEN>/status, /<JOB_TOKEN>/file/<name>) and prints the
     score files gzip+base64 encoded into the instance log, which the Vast API can fetch

    JOB_TOKEN=... python3 backend/news_llm_job.py
"""

import base64
import gzip
import http.server
import os
import shutil
import subprocess
import sys
import threading
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = "/root/job"
TOKEN = os.environ.get("JOB_TOKEN", "")
MODELS = [("Qwen/Qwen2.5-7B-Instruct", "qwen7b"),
          ("Qwen/Qwen2.5-32B-Instruct-AWQ", "qwen32b"),
          ("Qwen/Qwen2.5-72B-Instruct-AWQ", "qwen72b")]
if os.environ.get("JOB_MODELS"):                       # "repo:name,repo:name"
    MODELS = [tuple(m.split(":")) for m in os.environ["JOB_MODELS"].split(",")]
status = []
downloaded = {}


def log(msg: str):
    status.append(f"{time.strftime('%H:%M:%S')} {msg}")
    print(f"[job] {msg}", flush=True)


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes):
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not TOKEN or self.path != f"/{TOKEN}/posts":
            return self._send(404, b"")
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with open(os.path.join(WORK, "posts.jsonl"), "wb") as f:
            f.write(gzip.decompress(data))
        log(f"received posts over HTTP ({len(data)} bytes)")
        self._send(200, b"ok")

    def do_GET(self):
        if not TOKEN or not self.path.startswith(f"/{TOKEN}/"):
            return self._send(404, b"")
        if self.path == f"/{TOKEN}/status":
            return self._send(200, "\n".join(status).encode())
        path = os.path.join(WORK, os.path.basename(self.path))
        if self.path.startswith(f"/{TOKEN}/file/") and os.path.isfile(path):
            return self._send(200, open(path, "rb").read())
        self._send(404, b"")

    def log_message(self, *args):
        pass


def download_all():
    from huggingface_hub import snapshot_download
    for repo_id, name in MODELS:
        t0 = time.time()
        try:
            snapshot_download(repo_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
            downloaded[name] = True
            log(f"downloaded {repo_id} in {time.time() - t0:.0f} s")
        except Exception as e:
            downloaded[name] = False
            log(f"download failed {repo_id}: {e}")


def scrape_posts():
    log("no posts received; scraping @WatcherGuru")
    subprocess.run([sys.executable, "-m", "backend.telegram_data", "WatcherGuru"], cwd=REPO, check=True)
    subprocess.run([sys.executable, "-c", "from backend.news_llm_lab import prepare_posts; prepare_posts()"],
                   cwd=REPO, check=True)
    shutil.copy(os.path.join(REPO, "data", "news_llm", "posts.jsonl"), os.path.join(WORK, "posts.jsonl"))


def main():
    os.makedirs(WORK, exist_ok=True)
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", 8080), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log("http server on 8080")
    threading.Thread(target=download_all, daemon=True).start()
    posts = os.path.join(WORK, "posts.jsonl")
    deadline = time.time() + 240
    while not os.path.exists(posts) and time.time() < deadline:
        time.sleep(5)
    if not os.path.exists(posts):
        try:
            scrape_posts()
        except Exception as e:
            log(f"scrape failed: {e}")
            return
    n = sum(1 for _ in open(posts, encoding="utf-8"))
    log(f"{n} posts ready")
    for repo_id, name in MODELS:
        while name not in downloaded:
            time.sleep(5)
        if not downloaded[name]:
            continue
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(REPO, "backend", "news_llm_score.py"), posts, repo_id, name],
                           cwd=WORK, capture_output=True, text=True)
        out = os.path.join(WORK, f"scores_{name}.csv")
        if r.returncode != 0 or not os.path.exists(out):
            log(f"scoring failed {name}: {r.stderr[-1500:]}")
            continue
        log(f"scored {name} in {time.time() - t0:.0f} s")
        blob = base64.b64encode(gzip.compress(open(out, "rb").read())).decode()
        for i in range(0, len(blob), 400):                  # the Vast log keeps ~500 characters per line
            print(f"[b64 {name} {i // 400}] {blob[i:i + 400]}", flush=True)
        print(f"[b64 {name} end] {len(blob)}", flush=True)
    log("ALL DONE")
    time.sleep(3600)


if __name__ == "__main__":
    main()

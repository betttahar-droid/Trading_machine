"""
Telegram notifications for the trading plan.

Setup (once): in Telegram, message @BotFather, send /newbot, pick a name, and copy the token it gives you. Paste
it on the plan page and press Connect, then send any message to your new bot: the server reads your chat id from
that message and sends a test notification. Token and chat id are stored only on this PC, in
data/telegram_config.json (the data/ folder is never committed). TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID environment
variables override the file.

Messages are queued and sent from a background thread, so a slow or unreachable Telegram never holds up trading.
"""

import json
import logging
import os
import queue
import threading
import time
from typing import Optional

import requests

logger = logging.getLogger("layaquant.notifier")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CONFIG_FILE = os.path.join(DATA_DIR, "telegram_config.json")
API = "https://api.telegram.org/bot{token}/{method}"


class Notifier:
    def __init__(self):
        self.cfg = {"token": "", "chat_id": "", "enabled": True, "weekly": True, "daily": False}
        self._load()
        self._q: "queue.Queue[str]" = queue.Queue(maxsize=500)
        self._thread: Optional[threading.Thread] = None

    # ---------- config ----------
    def _load(self):
        if os.path.exists(CONFIG_FILE):
            try:
                self.cfg.update(json.load(open(CONFIG_FILE, encoding="utf-8")))
            except Exception as e:
                logger.warning(f"Could not read {CONFIG_FILE}: {e}")
        self.cfg["token"] = os.environ.get("TELEGRAM_BOT_TOKEN", self.cfg["token"])
        self.cfg["chat_id"] = os.environ.get("TELEGRAM_CHAT_ID", self.cfg["chat_id"])

    def _save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2)

    @property
    def ready(self) -> bool:
        return bool(self.cfg["token"] and self.cfg["chat_id"] and self.cfg["enabled"])

    def status(self) -> dict:
        tok = self.cfg["token"]
        return {"configured": bool(tok and self.cfg["chat_id"]), "enabled": self.cfg["enabled"],
                "weekly": self.cfg["weekly"], "daily": self.cfg["daily"],
                "token_hint": (tok.split(":")[0] + ":…" + tok[-4:]) if tok else "", "chat_id": self.cfg["chat_id"]}

    def settings(self, enabled: Optional[bool] = None, weekly: Optional[bool] = None, daily: Optional[bool] = None) -> dict:
        for k, v in (("enabled", enabled), ("weekly", weekly), ("daily", daily)):
            if v is not None:
                self.cfg[k] = bool(v)
        self._save()
        return self.status()

    def _call(self, method: str, token: Optional[str] = None, **params):
        r = requests.post(API.format(token=token or self.cfg["token"], method=method), json=params, timeout=15)
        return r.json()

    def connect(self, token: str, wait_s: int = 90) -> dict:
        """Check the token, then wait for the user to message the bot and remember that chat."""
        token = token.strip()
        try:
            me = self._call("getMe", token=token)
        except Exception as e:
            return {"status": "error", "message": f"Could not reach Telegram: {e}"}
        if not me.get("ok"):
            return {"status": "error", "message": "Telegram rejected this token. Copy it again from @BotFather."}
        bot = me["result"].get("username", "your bot")
        deadline = time.time() + wait_s
        offset = None
        while time.time() < deadline:
            try:
                upd = self._call("getUpdates", token=token, timeout=10, **({"offset": offset} if offset else {}))
            except Exception:
                time.sleep(2)
                continue
            for u in upd.get("result", []):
                offset = u["update_id"] + 1
                msg = u.get("message") or u.get("edited_message") or {}
                chat = msg.get("chat", {})
                if chat.get("id"):
                    self.cfg.update(token=token, chat_id=str(chat["id"]), enabled=True)
                    self._save()
                    self.send(f"✅ Connected. Trading-plan notifications will arrive here.")
                    return {"status": "connected", "bot": bot, "chat_id": self.cfg["chat_id"]}
        return {"status": "waiting_timeout", "bot": bot,
                "message": f"Token OK (@{bot}), but no message arrived. Open @{bot} in Telegram, press Start or send "
                           f"any text, then press Connect again."}

    # ---------- sending ----------
    def send(self, text: str):
        if not self.ready:
            return
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        try:
            self._q.put_nowait(text[:4000])
        except queue.Full:
            logger.warning("Telegram queue full; dropping a message")

    def _worker(self):
        while True:
            text = self._q.get()
            for attempt in range(3):
                try:
                    res = self._call("sendMessage", chat_id=self.cfg["chat_id"], text=text,
                                     disable_web_page_preview=True)
                    if res.get("ok"):
                        break
                    if res.get("error_code") == 429:           # rate limited
                        time.sleep(res.get("parameters", {}).get("retry_after", 3))
                    else:
                        logger.warning(f"Telegram send failed: {res.get('description')}")
                        break
                except Exception as e:
                    logger.warning(f"Telegram send error: {e}")
                    time.sleep(2 * (attempt + 1))
            time.sleep(1.1)                                        # stay under Telegram's per-chat limit


notifier = Notifier()

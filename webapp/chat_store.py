"""
chat_store.py — tiny JSON-file-backed chat history store.

No database needed: every chat (id, title, messages, timestamps) lives in
one JSON file on disk (chat_history.json, next to this script by default).
Writes are atomic (write to a temp file, then os.replace) and guarded by a
lock, so concurrent requests from the Flask dev server can't corrupt it.
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatStore:
    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            self._write({})

    # ── low-level I/O ────────────────────────────────────────────────
    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _write(self, data: dict) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)

    # ── public API ───────────────────────────────────────────────────
    def list_chats(self) -> list:
        with _LOCK:
            data = self._read()
        chats = [
            {
                "id": cid,
                "title": c.get("title", "New chat"),
                "created_at": c.get("created_at"),
                "updated_at": c.get("updated_at"),
                "message_count": len(c.get("messages", [])),
            }
            for cid, c in data.items()
        ]
        chats.sort(key=lambda c: c["updated_at"] or "", reverse=True)
        return chats

    def get_chat(self, chat_id: str):
        with _LOCK:
            data = self._read()
        return data.get(chat_id)

    def create_chat(self, title: str = "New chat") -> dict:
        with _LOCK:
            data = self._read()
            chat_id = uuid.uuid4().hex[:12]
            chat = {
                "id": chat_id,
                "title": title,
                "created_at": _now(),
                "updated_at": _now(),
                "messages": [],
            }
            data[chat_id] = chat
            self._write(data)
        return chat

    def delete_chat(self, chat_id: str) -> bool:
        with _LOCK:
            data = self._read()
            if chat_id not in data:
                return False
            del data[chat_id]
            self._write(data)
        return True

    def rename_chat(self, chat_id: str, title: str) -> bool:
        with _LOCK:
            data = self._read()
            if chat_id not in data:
                return False
            data[chat_id]["title"] = title
            data[chat_id]["updated_at"] = _now()
            self._write(data)
        return True

    def add_message(self, chat_id: str, role: str, content: str, sources=None) -> dict:
        """Appends a message to a chat, auto-titling the chat from the
        first user message if it's still on the default title."""
        with _LOCK:
            data = self._read()
            if chat_id not in data:
                raise KeyError(chat_id)
            chat = data[chat_id]
            message = {
                "role": role,
                "content": content,
                "sources": sources or [],
                "timestamp": _now(),
            }
            chat["messages"].append(message)
            chat["updated_at"] = _now()
            if role == "user" and chat.get("title", "New chat") == "New chat":
                title = content.strip().splitlines()[0][:60]
                chat["title"] = title if title else "New chat"
            self._write(data)
        return message

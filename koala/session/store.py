import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

from koala.session.models import SessionMeta


class SessionStore:
    """Persistent session store with JSONL message log and JSON metadata."""

    def __init__(self, session_dir: str):
        self._dir = Path(session_dir)
        self._messages_file = self._dir / "messages.jsonl"
        self._meta_file = self._dir / "meta.json"
        self._lock = asyncio.Lock()

    # --- Meta ---

    def _load_meta(self) -> Optional[SessionMeta]:
        if not self._meta_file.exists():
            return None
        with open(self._meta_file, "r") as f:
            return SessionMeta.from_dict(json.load(f))

    def _save_meta(self, meta: SessionMeta) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        with open(self._meta_file, "w") as f:
            json.dump(meta.to_dict(), f, indent=2, ensure_ascii=False)

    async def get_meta(self) -> Optional[SessionMeta]:
        async with self._lock:
            return self._load_meta()

    async def save_meta(self, meta: SessionMeta) -> None:
        async with self._lock:
            self._save_meta(meta)

    # --- Messages ---

    async def append_message(self, message: BaseMessage, meta: SessionMeta) -> None:
        async with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            line = json.dumps(
                {"timestamp": datetime.now().isoformat(), "message": message_to_dict(message)},
                ensure_ascii=False,
            )
            with open(self._messages_file, "a") as f:
                f.write(line + "\n")
            meta.message_count += 1
            meta.touch()
            self._save_meta(meta)

    async def append_messages(self, messages: list[BaseMessage], meta: SessionMeta) -> None:
        async with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            with open(self._messages_file, "a") as f:
                for msg in messages:
                    line = json.dumps(
                        {"timestamp": datetime.now().isoformat(), "message": message_to_dict(msg)},
                        ensure_ascii=False,
                    )
                    f.write(line + "\n")
            meta.message_count += len(messages)
            meta.touch()
            self._save_meta(meta)

    async def load_messages(self) -> list[BaseMessage]:
        async with self._lock:
            if not self._messages_file.exists():
                return []
            dicts = []
            with open(self._messages_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    dicts.append(entry["message"])
            return messages_from_dict(dicts)

    async def delete(self) -> None:
        async with self._lock:
            if self._dir.exists():
                for f in self._dir.iterdir():
                    f.unlink()
                self._dir.rmdir()

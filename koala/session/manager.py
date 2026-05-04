import uuid
from pathlib import Path
from typing import Optional

from langchain_core.messages import BaseMessage

from koala.context.memory import ShortTermMemory
from koala.session.models import SessionMeta
from koala.session.store import SessionStore


class SessionManager:
    """High-level session management API."""

    def __init__(self, base_dir: str = "sessions"):
        self._base_dir = Path(base_dir)

    def _store(self, session_id: str) -> SessionStore:
        return SessionStore(str(self._base_dir / session_id))

    @staticmethod
    def _new_session_id() -> str:
        return uuid.uuid4().hex[:12]

    # --- CRUD ---

    async def create_session(
        self,
        agent_config: dict | None = None,
        session_id: str | None = None,
    ) -> tuple[str, SessionStore]:
        sid = session_id or self._new_session_id()
        store = self._store(sid)
        meta = SessionMeta(session_id=sid, agent_config=agent_config or {})
        await store.save_meta(meta)
        return sid, store

    async def list_sessions(self) -> list[SessionMeta]:
        if not self._base_dir.exists():
            return []
        sessions = []
        for d in sorted(self._base_dir.iterdir()):
            if d.is_dir() and (d / "meta.json").exists():
                store = self._store(d.name)
                meta = await store.get_meta()
                if meta:
                    sessions.append(meta)
        return sessions

    async def get_session(self, session_id: str) -> Optional[tuple[SessionMeta, SessionStore]]:
        store = self._store(session_id)
        meta = await store.get_meta()
        if meta is None:
            return None
        return meta, store

    async def delete_session(self, session_id: str) -> bool:
        store = self._store(session_id)
        meta = await store.get_meta()
        if meta is None:
            return False
        await store.delete()
        return True

    # --- Memory integration ---

    async def save_messages(self, session_id: str, messages: list[BaseMessage]) -> None:
        store = self._store(session_id)
        meta = await store.get_meta()
        if meta is None:
            raise ValueError(f"Session {session_id} not found")
        await store.append_messages(messages, meta)

    async def save_memory(self, session_id: str, memory: ShortTermMemory) -> None:
        store = self._store(session_id)
        meta = await store.get_meta()
        if meta is None:
            raise ValueError(f"Session {session_id} not found")
        meta.message_count = 0
        await store.append_messages(memory.messages, meta)

    async def restore_memory(self, session_id: str, memory: ShortTermMemory) -> None:
        store = self._store(session_id)
        messages = await store.load_messages()
        memory.clear()
        memory.extend(messages)

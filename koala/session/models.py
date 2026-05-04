from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class SessionMeta:
    session_id: str
    created_at: str = ""
    updated_at: str = ""
    agent_config: dict = field(default_factory=dict)
    message_count: int = 0

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def touch(self):
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMeta":
        return cls(**data)

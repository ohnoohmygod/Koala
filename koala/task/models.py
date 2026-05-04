from enum import Enum
from pydantic import BaseModel


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Task(BaseModel):
    id: int
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[int] = []
    result: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "depends_on": self.depends_on,
            "result": self.result
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(**data)

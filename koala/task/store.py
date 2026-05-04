import asyncio
import json
from pathlib import Path
from typing import Optional

from koala.task.models import Task, TaskStatus


class TaskStore:
    """Persistent task store with JSON file backing."""

    def __init__(self, filepath: str):
        """Initialize the task store with a file path.

        Args:
            filepath: Path to the JSON file for persistence
        """
        self.filepath = Path(filepath)
        self._lock = asyncio.Lock()
        self._tasks: list[dict] = []

    def _load(self) -> list[dict]:
        """Load tasks from JSON file.

        Returns:
            List of task dictionaries, empty list if file doesn't exist
        """
        if not self.filepath.exists():
            return []

        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save(self, tasks: list[dict]) -> None:
        """Save tasks to JSON file.

        Args:
            tasks: List of task dictionaries to save
        """
        with open(self.filepath, 'w') as f:
            json.dump(tasks, f, indent=2)

    async def add_task(
        self,
        name: str,
        description: str,
        depends_on: list[int] | None = None
    ) -> Task:
        """Create a new task with auto-incremented ID.

        Args:
            name: Task name
            description: Task description
            depends_on: Optional list of task IDs this task depends on

        Returns:
            The created Task object
        """
        async with self._lock:
            tasks = self._load()

            # Auto-increment ID (max existing ID + 1, or 1 if empty)
            if tasks:
                next_id = max(task["id"] for task in tasks) + 1
            else:
                next_id = 1

            task_data = {
                "id": next_id,
                "name": name,
                "description": description,
                "status": TaskStatus.PENDING,
                "depends_on": depends_on or [],
                "result": None
            }

            tasks.append(task_data)
            self._save(tasks)

            return Task(**task_data)

    async def get_task(self, task_id: int) -> Optional[Task]:
        """Retrieve a task by ID.

        Args:
            task_id: The task ID to retrieve

        Returns:
            Task object if found, None otherwise
        """
        async with self._lock:
            tasks = self._load()
            for task_data in tasks:
                if task_data["id"] == task_id:
                    return Task(**task_data)
            return None

    async def update_task(self, task_id: int, **kwargs) -> Optional[Task]:
        """Update specific fields of a task.

        Args:
            task_id: The task ID to update
            **kwargs: Fields to update (e.g., status, result)

        Returns:
            Updated Task object if found, None otherwise
        """
        async with self._lock:
            tasks = self._load()
            for i, task_data in enumerate(tasks):
                if task_data["id"] == task_id:
                    # Update specified fields
                    for key, value in kwargs.items():
                        if key in task_data:
                            task_data[key] = value
                    tasks[i] = task_data
                    self._save(tasks)
                    return Task(**task_data)
            return None

    async def list_tasks(
        self,
        status: TaskStatus | None = None
    ) -> list[Task]:
        """List all tasks, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of Task objects
        """
        async with self._lock:
            tasks = self._load()
            result = [Task(**task_data) for task_data in tasks]

            if status is not None:
                result = [t for t in result if t.status == status]

            return result

    async def get_ready_tasks(self) -> list[Task]:
        """Get tasks that are ready to run.

        A task is ready if:
        - Its status is PENDING
        - All its dependencies have status COMPLETED

        Returns:
            List of ready Task objects
        """
        async with self._lock:
            tasks = self._load()
            all_tasks = {task["id"]: task for task in tasks}

            ready_tasks = []
            for task_data in tasks:
                # Skip non-pending tasks
                if task_data["status"] != TaskStatus.PENDING:
                    continue

                # Check if all dependencies are completed
                dependencies_met = all(
                    dep_id in all_tasks and
                    all_tasks[dep_id]["status"] == TaskStatus.COMPLETED
                    for dep_id in task_data["depends_on"]
                )

                if dependencies_met:
                    ready_tasks.append(Task(**task_data))

            return ready_tasks

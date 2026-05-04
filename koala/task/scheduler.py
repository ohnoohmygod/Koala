import asyncio
from logging import getLogger
from typing import Callable, Awaitable

from koala.task.models import Task, TaskStatus
from koala.task.notify import TaskNotifier
from koala.task.store import TaskStore


logger = getLogger(__name__)


class TaskScheduler:
    """Task scheduler that manages task execution and dependency resolution."""

    def __init__(self, store: TaskStore, notifier: TaskNotifier):
        """Initialize the scheduler with dependencies.

        Args:
            store: TaskStore for task persistence
            notifier: TaskNotifier for completion notifications
        """
        self._store = store
        self._notifier = notifier

    async def dispatch_ready_tasks(
        self,
        executor: Callable[[Task], Awaitable[str]]
    ) -> list[asyncio.Task]:
        """Dispatch all ready tasks to the executor.

        Args:
            executor: Async function that takes a Task and returns a result string

        Returns:
            List of asyncio.Task objects for the dispatched tasks
        """
        ready_tasks = await self._store.get_ready_tasks()
        dispatched = []

        for task in ready_tasks:
            # Mark task as RUNNING
            await self._store.update_task(task.id, status=TaskStatus.RUNNING)

            # Create a wrapper that handles completion/failure
            async def run_with_error_handling(task: Task = task):
                try:
                    result = await executor(task)
                    await self.on_task_complete(task.id, result)
                except Exception as e:
                    await self.on_task_failed(task.id, str(e))

            # Start the task
            dispatched.append(asyncio.create_task(run_with_error_handling()))

        return dispatched

    async def on_task_complete(self, task_id: int, result: str) -> list[Task]:
        """Handle task completion.

        Args:
            task_id: The ID of the completed task
            result: The result string from the task

        Returns:
            List of newly unlocked tasks (ready to run)
        """
        # Update task status
        await self._store.update_task(task_id, status=TaskStatus.COMPLETED, result=result)

        # Enqueue notification
        await self._notifier.enqueue(task_id, result)

        # Find and return newly unlocked tasks
        return await self.get_newly_unlocked(task_id)

    async def on_task_failed(self, task_id: int, error: str) -> None:
        """Handle task failure.

        Args:
            task_id: The ID of the failed task
            error: The error message
        """
        await self._store.update_task(task_id, status=TaskStatus.FAILED, result=error)
        logger.error(f"Task {task_id} failed: {error}")

    async def get_newly_unlocked(self, task_id: int) -> list[Task]:
        """Find tasks that are newly unlocked by the completion of task_id.

        A task is unlocked if:
        - It depends on task_id
        - It is still PENDING
        - ALL its dependencies are now COMPLETED

        Args:
            task_id: The ID of the just-completed task

        Returns:
            List of newly unlocked Task objects
        """
        all_tasks = await self._store.list_tasks()
        newly_unlocked = []

        # Build a map of task status for quick lookup
        task_statuses = {task.id: task.status for task in all_tasks}

        for task in all_tasks:
            # Skip if not dependent on this task or not pending
            if task_id not in task.depends_on:
                continue
            if task.status != TaskStatus.PENDING:
                continue

            # Check if all dependencies are completed
            all_deps_complete = all(
                dep_id in task_statuses and task_statuses[dep_id] == TaskStatus.COMPLETED
                for dep_id in task.depends_on
            )

            if all_deps_complete:
                newly_unlocked.append(task)

        return newly_unlocked

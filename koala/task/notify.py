import asyncio
from typing import Tuple


class TaskNotifier:
    """Notification queue for completed tasks."""

    def __init__(self) -> None:
        """Initialize an asyncio queue for notifications."""
        self._queue: asyncio.Queue[Tuple[int, str]] = asyncio.Queue()

    async def enqueue(self, task_id: int, result: str) -> None:
        """Enqueue a notification when a subagent completes a task.

        Args:
            task_id: The ID of the completed task
            result: The result message from the task
        """
        await self._queue.put((task_id, result))

    async def drain(self) -> list[Tuple[int, str]]:
        """Drain all pending notifications from the queue.

        Returns:
            List of (task_id, result) tuples for all pending notifications
        """
        notifications = []

        while not self._queue.empty():
            try:
                notification = self._queue.get_nowait()
                notifications.append(notification)
            except asyncio.QueueEmpty:
                break

        return notifications

    async def wait(self) -> Tuple[int, str]:
        """Asynchronously wait for the next notification.

        This method blocks until a notification is available, making it
        suitable for event loop competition patterns.

        Returns:
            A tuple of (task_id, result) for the next notification
        """
        return await self._queue.get()

    def format_notification(self, task_id: int, result: str) -> str:
        """Format a notification as a human-readable string.

        Args:
            task_id: The ID of the completed task
            result: The result message

        Returns:
            Formatted notification string like "[任务 T1 完成] 结果：xxx"
        """
        return f"[任务 T{task_id} 完成] 结果：{result}"

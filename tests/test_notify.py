import asyncio
import pytest

from koala.task.notify import TaskNotifier


class TestTaskNotifierBasics:
    """Test basic TaskNotifier functionality."""

    def test_enqueue_and_drain(self):
        """Test enqueueing multiple notifications and draining them."""
        notifier = TaskNotifier()

        # Enqueue multiple notifications
        asyncio.run(notifier.enqueue(1, "Task 1 completed"))
        asyncio.run(notifier.enqueue(2, "Task 2 completed"))
        asyncio.run(notifier.enqueue(3, "Task 3 completed"))

        # Drain and verify
        result = asyncio.run(notifier.drain())
        assert result == [
            (1, "Task 1 completed"),
            (2, "Task 2 completed"),
            (3, "Task 3 completed")
        ]

    def test_drain_empty_queue(self):
        """Test draining an empty queue returns empty list."""
        notifier = TaskNotifier()
        result = asyncio.run(notifier.drain())
        assert result == []

    def test_drain_after_enqueue_depletes_queue(self):
        """Test that draining empties the queue."""
        notifier = TaskNotifier()

        asyncio.run(notifier.enqueue(1, "Task completed"))

        # First drain should return the notification
        result1 = asyncio.run(notifier.drain())
        assert result1 == [(1, "Task completed")]

        # Second drain should return empty
        result2 = asyncio.run(notifier.drain())
        assert result2 == []


class TestTaskNotifierWait:
    """Test async wait functionality."""

    def test_wait_returns_notification(self):
        """Test that wait returns a single notification."""
        notifier = TaskNotifier()

        async def test_workflow():
            # Enqueue in background
            task = asyncio.create_task(notifier.wait())

            # Give wait a chance to start waiting
            await asyncio.sleep(0.01)

            # Enqueue notification
            await notifier.enqueue(1, "Task completed")

            # Wait should return the notification
            result = await task
            return result

        result = asyncio.run(test_workflow())
        assert result == (1, "Task completed")

    def test_wait_with_multiple_notifications(self):
        """Test wait returns first notification when multiple are queued."""
        notifier = TaskNotifier()

        async def test_workflow():
            # Enqueue first
            await notifier.enqueue(1, "First")
            await notifier.enqueue(2, "Second")

            # Wait should return the first one
            result = await notifier.wait()
            return result

        result = asyncio.run(test_workflow())
        assert result == (1, "First")

    def test_multiple_waits_get_different_notifications(self):
        """Test that multiple wait calls return different notifications."""
        notifier = TaskNotifier()

        async def test_workflow():
            # Enqueue two notifications
            await notifier.enqueue(1, "First")
            await notifier.enqueue(2, "Second")

            # Two waits should get different notifications
            first = await notifier.wait()
            second = await notifier.wait()

            return first, second

        first, second = asyncio.run(test_workflow())
        assert first == (1, "First")
        assert second == (2, "Second")


class TestTaskNotifierFormat:
    """Test notification formatting."""

    def test_format_notification(self):
        """Test formatting notification string."""
        notifier = TaskNotifier()
        result = notifier.format_notification(1, "Task completed successfully")
        expected = "[任务 T1 完成] 结果：Task completed successfully"
        assert result == expected

    def test_format_notification_with_different_ids(self):
        """Test formatting with various task IDs."""
        notifier = TaskNotifier()

        assert notifier.format_notification(1, "Done") == "[任务 T1 完成] 结果：Done"
        assert notifier.format_notification(42, "Finished") == "[任务 T42 完成] 结果：Finished"
        assert notifier.format_notification(100, "Complete") == "[任务 T100 完成] 结果：Complete"


class TestTaskNotifierConcurrency:
    """Test concurrent operations."""

    def test_concurrent_enqueue(self):
        """Test multiple coroutines enqueuing concurrently."""
        notifier = TaskNotifier()

        async def enqueue_many():
            tasks = [
                notifier.enqueue(i, f"Task {i}")
                for i in range(1, 11)
            ]
            await asyncio.gather(*tasks)

        asyncio.run(enqueue_many())

        # Drain and verify all notifications
        result = asyncio.run(notifier.drain())
        assert len(result) == 10
        assert set(result) == {
            (1, "Task 1"), (2, "Task 2"), (3, "Task 3"), (4, "Task 4"),
            (5, "Task 5"), (6, "Task 6"), (7, "Task 7"), (8, "Task 8"),
            (9, "Task 9"), (10, "Task 10")
        }

    def test_concurrent_enqueue_and_drain(self):
        """Test concurrent enqueue and drain operations."""
        notifier = TaskNotifier()

        async def concurrent_operations():
            # Create tasks for concurrent execution
            results = await asyncio.gather(
                notifier.enqueue(1, "Task 1"),
                notifier.enqueue(2, "Task 2"),
                notifier.enqueue(3, "Task 3"),
                notifier.drain(),
            )

            # The last result should be the drain result
            return results[-1]

        result = asyncio.run(concurrent_operations())

        # Drain should have collected all three notifications
        assert len(result) == 3
        # Order may vary due to concurrency
        assert set(result) == {
            (1, "Task 1"),
            (2, "Task 2"),
            (3, "Task 3")
        }

    def test_concurrent_waits(self):
        """Test multiple waiters competing for notifications."""
        notifier = TaskNotifier()

        async def waiters_competing():
            # Start multiple waiters
            waiters = [
                asyncio.create_task(notifier.wait())
                for _ in range(3)
            ]

            # Give waiters time to start waiting
            await asyncio.sleep(0.01)

            # Enqueue notifications
            await notifier.enqueue(1, "First")
            await notifier.enqueue(2, "Second")
            await notifier.enqueue(3, "Third")

            # Wait for all waiters to complete
            results = await asyncio.gather(*waiters)
            return results

        results = asyncio.run(waiters_competing())

        # All waiters should have gotten a notification
        assert len(results) == 3
        assert set(results) == {
            (1, "First"),
            (2, "Second"),
            (3, "Third")
        }

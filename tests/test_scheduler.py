import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from koala.task.models import Task, TaskStatus
from koala.task.notify import TaskNotifier
from koala.task.scheduler import TaskScheduler
from koala.task.store import TaskStore


@pytest.fixture
def mock_store():
    """Create a mock TaskStore."""
    store = AsyncMock(spec=TaskStore)
    return store


@pytest.fixture
def mock_notifier():
    """Create a mock TaskNotifier."""
    notifier = AsyncMock(spec=TaskNotifier)
    return notifier


@pytest.fixture
def scheduler(mock_store, mock_notifier):
    """Create a TaskScheduler with mocked dependencies."""
    return TaskScheduler(mock_store, mock_notifier)


@pytest.mark.asyncio
async def test_dispatch_ready_tasks_no_dependencies(scheduler, mock_store):
    """Test that tasks without dependencies are immediately executed."""
    # Setup: Create tasks with no dependencies
    task1 = Task(id=1, name="Task 1", description="First task", status=TaskStatus.PENDING, depends_on=[])
    task2 = Task(id=2, name="Task 2", description="Second task", status=TaskStatus.PENDING, depends_on=[])

    mock_store.get_ready_tasks.return_value = [task1, task2]
    mock_store.update_task.return_value = None

    # Mock executor function
    async def mock_executor(task: Task):
        return f"Completed {task.name}"

    # Dispatch tasks
    dispatched = await scheduler.dispatch_ready_tasks(mock_executor)

    # Verify both tasks were marked as RUNNING
    assert mock_store.update_task.call_count == 2
    mock_store.update_task.assert_any_call(1, status=TaskStatus.RUNNING)
    mock_store.update_task.assert_any_call(2, status=TaskStatus.RUNNING)

    # Verify tasks were dispatched
    assert len(dispatched) == 2


@pytest.mark.asyncio
async def test_dispatch_ready_tasks_with_pending_dependencies(scheduler, mock_store):
    """Test that tasks with unmet dependencies are not executed."""
    # Setup: Task depends on task_id=1 which is not completed yet
    task_with_dep = Task(
        id=2,
        name="Task 2",
        description="Dependent task",
        status=TaskStatus.PENDING,
        depends_on=[1]
    )

    mock_store.get_ready_tasks.return_value = []  # Not ready due to dependency
    mock_store.update_task.return_value = None

    async def mock_executor(task: Task):
        return f"Completed {task.name}"

    # Dispatch tasks
    dispatched = await scheduler.dispatch_ready_tasks(mock_executor)

    # Verify no tasks were dispatched or updated
    assert len(dispatched) == 0
    mock_store.update_task.assert_not_called()


@pytest.mark.asyncio
async def test_on_task_complete(scheduler, mock_store, mock_notifier):
    """Test that task completion updates status, sends notification, and returns unlocked tasks."""
    # Setup: Task 1 completes, Task 2 depends on Task 1
    task1 = Task(
        id=1,
        name="Task 1",
        description="Completed task",
        status=TaskStatus.COMPLETED,
        depends_on=[]
    )
    task2 = Task(
        id=2,
        name="Task 2",
        description="Dependent task",
        status=TaskStatus.PENDING,
        depends_on=[1]
    )

    mock_store.update_task.return_value = None
    mock_store.list_tasks.return_value = [task1, task2]

    # Complete task 1
    result = "Task completed successfully"
    unlocked = await scheduler.on_task_complete(1, result)

    # Verify status updated to COMPLETED
    mock_store.update_task.assert_any_call(1, status=TaskStatus.COMPLETED, result=result)

    # Verify notification was enqueued
    mock_notifier.enqueue.assert_called_once_with(1, result)

    # Verify dependent task is returned as unlocked
    assert len(unlocked) == 1
    assert unlocked[0].id == 2


@pytest.mark.asyncio
async def test_on_task_failed(scheduler, mock_store):
    """Test that task failure is properly recorded."""
    mock_store.update_task.return_value = None

    error_msg = "Task failed with error"
    await scheduler.on_task_failed(1, error_msg)

    # Verify status updated to FAILED
    mock_store.update_task.assert_called_once_with(1, status=TaskStatus.FAILED, result=error_msg)


@pytest.mark.asyncio
async def test_get_newly_unlocked_single_dependency(scheduler, mock_store):
    """Test that a task is unlocked when its single dependency completes."""
    # Task 2 depends on Task 1 (which just completed)
    task2 = Task(
        id=2,
        name="Task 2",
        description="Dependent task",
        status=TaskStatus.PENDING,
        depends_on=[1]
    )

    # Task 1 is now COMPLETED
    task1 = Task(
        id=1,
        name="Task 1",
        description="Completed task",
        status=TaskStatus.COMPLETED,
        depends_on=[]
    )

    mock_store.list_tasks.return_value = [task2, task1]

    # Get newly unlocked tasks after task 1 completes
    unlocked = await scheduler.get_newly_unlocked(1)

    # Task 2 should be unlocked
    assert len(unlocked) == 1
    assert unlocked[0].id == 2


@pytest.mark.asyncio
async def test_get_newly_unlocked_multiple_dependencies_partial(scheduler, mock_store):
    """Test that a task with multiple dependencies is not unlocked until all are complete."""
    # Task 3 depends on both Task 1 and Task 2
    task3 = Task(
        id=3,
        name="Task 3",
        description="Multi-dependency task",
        status=TaskStatus.PENDING,
        depends_on=[1, 2]
    )

    # Only Task 1 is COMPLETED, Task 2 is still PENDING
    task1 = Task(
        id=1,
        name="Task 1",
        description="First completed task",
        status=TaskStatus.COMPLETED,
        depends_on=[]
    )
    task2 = Task(
        id=2,
        name="Task 2",
        description="Second pending task",
        status=TaskStatus.PENDING,
        depends_on=[]
    )

    mock_store.list_tasks.return_value = [task3, task1, task2]

    # Get newly unlocked after only task 1 completes
    unlocked = await scheduler.get_newly_unlocked(1)

    # Task 3 should NOT be unlocked (task 2 still pending)
    assert len(unlocked) == 0


@pytest.mark.asyncio
async def test_get_newly_unlocked_multiple_dependencies_all_complete(scheduler, mock_store):
    """Test that a task with multiple dependencies is unlocked when all are complete."""
    # Task 3 depends on both Task 1 and Task 2
    task3 = Task(
        id=3,
        name="Task 3",
        description="Multi-dependency task",
        status=TaskStatus.PENDING,
        depends_on=[1, 2]
    )

    # Both Task 1 and Task 2 are COMPLETED
    task1 = Task(
        id=1,
        name="Task 1",
        description="First completed task",
        status=TaskStatus.COMPLETED,
        depends_on=[]
    )
    task2 = Task(
        id=2,
        name="Task 2",
        description="Second completed task",
        status=TaskStatus.COMPLETED,
        depends_on=[]
    )

    mock_store.list_tasks.return_value = [task3, task1, task2]

    # Get newly unlocked after task 2 completes (task 1 already complete)
    unlocked = await scheduler.get_newly_unlocked(2)

    # Task 3 should be unlocked now
    assert len(unlocked) == 1
    assert unlocked[0].id == 3


@pytest.mark.asyncio
async def test_executor_exception_calls_on_task_failed(scheduler, mock_store):
    """Test that executor exceptions trigger on_task_failed."""
    task1 = Task(id=1, name="Task 1", description="Failing task", status=TaskStatus.PENDING, depends_on=[])

    mock_store.get_ready_tasks.return_value = [task1]
    mock_store.update_task.return_value = None

    # Mock executor that raises an exception
    async def failing_executor(task: Task):
        raise ValueError("Executor error")

    # Mock on_task_failed to track calls
    scheduler.on_task_failed = AsyncMock()

    # Dispatch should handle the exception
    await scheduler.dispatch_ready_tasks(failing_executor)

    # Wait a bit for async tasks to complete
    await asyncio.sleep(0.1)

    # Verify on_task_failed was called
    scheduler.on_task_failed.assert_called_once()
    call_args = scheduler.on_task_failed.call_args
    assert call_args[0][0] == 1  # task_id
    assert "Executor error" in call_args[0][1]  # error message

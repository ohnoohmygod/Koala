import json
import pytest
import asyncio
from pathlib import Path

from koala.task.models import Task, TaskStatus
from koala.task.store import TaskStore


@pytest.fixture
def temp_store(tmp_path):
    """Create a TaskStore with a temporary file path."""
    filepath = tmp_path / "tasks.json"
    return TaskStore(str(filepath))


@pytest.mark.asyncio
async def test_add_task_auto_increment(temp_store):
    """Test that task IDs auto-increment starting from 1."""
    task1 = await temp_store.add_task("Task 1", "First task")
    assert task1.id == 1
    assert task1.name == "Task 1"
    assert task1.description == "First task"
    assert task1.status == TaskStatus.PENDING

    task2 = await temp_store.add_task("Task 2", "Second task")
    assert task2.id == 2
    assert task2.name == "Task 2"


@pytest.mark.asyncio
async def test_add_task_with_depends_on(temp_store):
    """Test creating a task with dependencies."""
    task1 = await temp_store.add_task("Task 1", "First task")
    task2 = await temp_store.add_task("Task 2", "Second task", depends_on=[task1.id])

    assert task2.depends_on == [1]
    assert task2.status == TaskStatus.PENDING


@pytest.mark.asyncio
async def test_get_task(temp_store):
    """Test retrieving a task by ID."""
    created = await temp_store.add_task("Test Task", "Description")
    retrieved = await temp_store.get_task(created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.name == "Test Task"
    assert retrieved.description == "Description"


@pytest.mark.asyncio
async def test_get_task_not_found(temp_store):
    """Test retrieving a non-existent task returns None."""
    result = await temp_store.get_task(999)
    assert result is None


@pytest.mark.asyncio
async def test_update_task_status(temp_store):
    """Test updating task status."""
    task = await temp_store.add_task("Test Task", "Description")

    updated = await temp_store.update_task(task.id, status=TaskStatus.RUNNING)
    assert updated is not None
    assert updated.status == TaskStatus.RUNNING
    assert updated.id == task.id

    # Verify persistence
    retrieved = await temp_store.get_task(task.id)
    assert retrieved.status == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_update_task_result(temp_store):
    """Test updating task result."""
    task = await temp_store.add_task("Test Task", "Description")

    updated = await temp_store.update_task(task.id, result="Task completed successfully")
    assert updated is not None
    assert updated.result == "Task completed successfully"


@pytest.mark.asyncio
async def test_update_task_multiple_fields(temp_store):
    """Test updating multiple fields at once."""
    task = await temp_store.add_task("Test Task", "Description")

    updated = await temp_store.update_task(
        task.id,
        status=TaskStatus.COMPLETED,
        result="Done"
    )
    assert updated.status == TaskStatus.COMPLETED
    assert updated.result == "Done"


@pytest.mark.asyncio
async def test_update_task_not_found(temp_store):
    """Test updating a non-existent task returns None."""
    result = await temp_store.update_task(999, status=TaskStatus.RUNNING)
    assert result is None


@pytest.mark.asyncio
async def test_list_tasks_all(temp_store):
    """Test listing all tasks."""
    await temp_store.add_task("Task 1", "First")
    await temp_store.add_task("Task 2", "Second")
    await temp_store.add_task("Task 3", "Third")

    tasks = await temp_store.list_tasks()
    assert len(tasks) == 3
    assert [t.name for t in tasks] == ["Task 1", "Task 2", "Task 3"]


@pytest.mark.asyncio
async def test_list_tasks_filter_by_status(temp_store):
    """Test listing tasks filtered by status."""
    task1 = await temp_store.add_task("Task 1", "First")
    task2 = await temp_store.add_task("Task 2", "Second")

    await temp_store.update_task(task1.id, status=TaskStatus.COMPLETED)
    await temp_store.update_task(task2.id, status=TaskStatus.RUNNING)

    pending_tasks = await temp_store.list_tasks(status=TaskStatus.PENDING)
    assert len(pending_tasks) == 0

    completed_tasks = await temp_store.list_tasks(status=TaskStatus.COMPLETED)
    assert len(completed_tasks) == 1
    assert completed_tasks[0].id == task1.id


@pytest.mark.asyncio
async def test_get_ready_tasks_no_dependencies(temp_store):
    """Test that tasks with no dependencies are ready."""
    task1 = await temp_store.add_task("Task 1", "First task")
    task2 = await temp_store.add_task("Task 2", "Second task")

    ready_tasks = await temp_store.get_ready_tasks()
    assert len(ready_tasks) == 2
    assert {t.id for t in ready_tasks} == {task1.id, task2.id}


@pytest.mark.asyncio
async def test_get_ready_tasks_with_unmet_dependencies(temp_store):
    """Test that tasks with unmet dependencies are not ready."""
    task1 = await temp_store.add_task("Task 1", "First task")
    task2 = await temp_store.add_task("Task 2", "Second task", depends_on=[task1.id])

    ready_tasks = await temp_store.get_ready_tasks()
    assert len(ready_tasks) == 1
    assert ready_tasks[0].id == task1.id


@pytest.mark.asyncio
async def test_get_ready_tasks_with_met_dependencies(temp_store):
    """Test that tasks become ready when dependencies are completed."""
    task1 = await temp_store.add_task("Task 1", "First task")
    task2 = await temp_store.add_task("Task 2", "Second task", depends_on=[task1.id])

    # Initially only task1 is ready
    ready_tasks = await temp_store.get_ready_tasks()
    assert len(ready_tasks) == 1

    # Complete task1
    await temp_store.update_task(task1.id, status=TaskStatus.COMPLETED)

    # Now task2 should also be ready
    ready_tasks = await temp_store.get_ready_tasks()
    assert len(ready_tasks) == 1
    assert ready_tasks[0].id == task2.id


@pytest.mark.asyncio
async def test_get_ready_tasks_multiple_dependencies(temp_store):
    """Test ready tasks with multiple dependencies."""
    task1 = await temp_store.add_task("Task 1", "First task")
    task2 = await temp_store.add_task("Task 2", "Second task")
    task3 = await temp_store.add_task("Task 3", "Third task", depends_on=[task1.id, task2.id])

    # Initially only task1 and task2 are ready
    ready_tasks = await temp_store.get_ready_tasks()
    assert len(ready_tasks) == 2

    # Complete only task1
    await temp_store.update_task(task1.id, status=TaskStatus.COMPLETED)

    # task3 still not ready
    ready_tasks = await temp_store.get_ready_tasks()
    assert len(ready_tasks) == 1
    assert ready_tasks[0].id == task2.id

    # Complete task2
    await temp_store.update_task(task2.id, status=TaskStatus.COMPLETED)

    # Now task3 is ready
    ready_tasks = await temp_store.get_ready_tasks()
    assert len(ready_tasks) == 1
    assert ready_tasks[0].id == task3.id


@pytest.mark.asyncio
async def test_get_ready_tasks_only_considers_completed(temp_store):
    """Test that only COMPLETED status satisfies dependencies."""
    task1 = await temp_store.add_task("Task 1", "First task")
    task2 = await temp_store.add_task("Task 2", "Second task", depends_on=[task1.id])

    # Mark task1 as RUNNING (not COMPLETED)
    await temp_store.update_task(task1.id, status=TaskStatus.RUNNING)

    # task2 should NOT be ready
    ready_tasks = await temp_store.get_ready_tasks()
    assert len(ready_tasks) == 0


@pytest.mark.asyncio
async def test_persistence_to_file(temp_store, tmp_path):
    """Test that tasks are persisted to JSON file."""
    filepath = tmp_path / "tasks.json"

    task1 = await temp_store.add_task("Task 1", "First task")
    await temp_store.update_task(task1.id, status=TaskStatus.COMPLETED, result="Done")

    # Verify file was created and contains correct data
    assert filepath.exists()

    with open(filepath, 'r') as f:
        data = json.load(f)

    assert len(data) == 1
    assert data[0]["id"] == 1
    assert data[0]["name"] == "Task 1"
    assert data[0]["description"] == "First task"
    assert data[0]["status"] == "COMPLETED"
    assert data[0]["result"] == "Done"


@pytest.mark.asyncio
async def test_load_from_existing_file(tmp_path):
    """Test that existing tasks are loaded from file."""
    filepath = tmp_path / "tasks.json"

    # Create pre-existing data
    existing_data = [
        {
            "id": 1,
            "name": "Existing Task",
            "description": "Already exists",
            "status": "COMPLETED",
            "depends_on": [],
            "result": "Finished"
        }
    ]

    with open(filepath, 'w') as f:
        json.dump(existing_data, f)

    # Create store and verify it loads existing data
    store = TaskStore(str(filepath))
    tasks = await store.list_tasks()

    assert len(tasks) == 1
    assert tasks[0].name == "Existing Task"
    assert tasks[0].status == TaskStatus.COMPLETED

    # New task should get ID 2 (auto-increment from existing)
    new_task = await store.add_task("New Task", "New description")
    assert new_task.id == 2


@pytest.mark.asyncio
async def test_empty_file_initialization(tmp_path):
    """Test that store initializes correctly when file doesn't exist."""
    filepath = tmp_path / "nonexistent.json"

    store = TaskStore(str(filepath))
    tasks = await store.list_tasks()

    assert len(tasks) == 0
    assert not filepath.exists()  # File shouldn't be created until first save


@pytest.mark.asyncio
async def test_concurrent_operations(temp_store):
    """Test that concurrent operations are thread-safe."""
    tasks = []

    # Create multiple tasks concurrently
    async def create_task(name):
        return await temp_store.add_task(name, f"Description for {name}")

    # Create 10 tasks concurrently
    tasks = await asyncio.gather(*[
        create_task(f"Task {i}") for i in range(10)
    ])

    # Verify all tasks were created with unique IDs
    assert len(tasks) == 10
    ids = [t.id for t in tasks]
    assert sorted(ids) == list(range(1, 11))

    # Verify all tasks can be retrieved
    for task in tasks:
        retrieved = await temp_store.get_task(task.id)
        assert retrieved is not None
        assert retrieved.name == task.name


@pytest.mark.asyncio
async def test_concurrent_updates(temp_store):
    """Test that concurrent updates to the same task are safe."""
    task = await temp_store.add_task("Test Task", "Description")

    # Perform multiple updates concurrently
    async def update_status(status):
        await temp_store.update_task(task.id, status=status)

    await asyncio.gather(*[
        update_status(TaskStatus.RUNNING),
        update_status(TaskStatus.COMPLETED),
    ])

    # Final state should be consistent
    retrieved = await temp_store.get_task(task.id)
    assert retrieved.status in [TaskStatus.RUNNING, TaskStatus.COMPLETED]


@pytest.mark.asyncio
async def test_failed_task_not_ready(temp_store):
    """Test that FAILED tasks don't make dependents ready."""
    task1 = await temp_store.add_task("Task 1", "First task")
    task2 = await temp_store.add_task("Task 2", "Second task", depends_on=[task1.id])

    # Mark task1 as FAILED
    await temp_store.update_task(task1.id, status=TaskStatus.FAILED)

    # task2 should NOT be ready (only COMPLETED counts)
    ready_tasks = await temp_store.get_ready_tasks()
    assert len(ready_tasks) == 0

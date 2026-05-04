import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage

from koala.task.models import Task, TaskStatus
from koala.llm import LLMClient


def _make_mock_llm():
    return MagicMock(spec=LLMClient)


def _make_mock_store():
    """Create a mock TaskStore with common methods."""
    store = MagicMock()
    store.add_task = AsyncMock()
    store.get_task = AsyncMock()
    store.update_task = AsyncMock()
    store.list_tasks = AsyncMock()
    return store


def _make_mock_runner():
    """Create a mock AgentRunner with register_background_task method."""
    runner = MagicMock()
    runner.register_background_task = AsyncMock()
    return runner


class TestCreateTaskTool:
    @pytest.mark.asyncio
    async def test_create_task_with_name_and_description(self):
        """Test creating a task with just name and description."""
        mock_store = _make_mock_store()
        mock_store.add_task.return_value = Task(
            id=1,
            name="Test Task",
            description="A test task",
            status=TaskStatus.PENDING,
            depends_on=[],
            result=None
        )

        from koala.tools.task_tools import CreateTaskTool
        tool = CreateTaskTool(store=mock_store)

        result = await tool.ainvoke({
            "name": "Test Task",
            "description": "A test task"
        })

        mock_store.add_task.assert_called_once_with(
            name="Test Task",
            description="A test task",
            depends_on=[]
        )
        assert "id" in result
        assert result["id"] == 1
        assert result["name"] == "Test Task"
        assert result["status"] == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_create_task_with_dependencies(self):
        """Test creating a task with dependencies."""
        mock_store = _make_mock_store()
        mock_store.add_task.return_value = Task(
            id=2,
            name="Dependent Task",
            description="Depends on task 1",
            status=TaskStatus.PENDING,
            depends_on=[1],
            result=None
        )

        from koala.tools.task_tools import CreateTaskTool
        tool = CreateTaskTool(store=mock_store)

        result = await tool.ainvoke({
            "name": "Dependent Task",
            "description": "Depends on task 1",
            "depends_on": [1]
        })

        mock_store.add_task.assert_called_once_with(
            name="Dependent Task",
            description="Depends on task 1",
            depends_on=[1]
        )
        assert result["depends_on"] == [1]

    @pytest.mark.asyncio
    async def test_create_task_returns_dict(self):
        """Test that create_task returns a dict with all task fields."""
        mock_store = _make_mock_store()
        mock_store.add_task.return_value = Task(
            id=1,
            name="Test",
            description="Test desc",
            status=TaskStatus.PENDING,
            depends_on=[],
            result=None
        )

        from koala.tools.task_tools import CreateTaskTool
        tool = CreateTaskTool(store=mock_store)

        result = await tool.ainvoke({
            "name": "Test",
            "description": "Test desc"
        })

        assert isinstance(result, dict)
        assert "id" in result
        assert "name" in result
        assert "description" in result
        assert "status" in result
        assert "depends_on" in result
        assert "result" in result


class TestQueryTaskTool:
    @pytest.mark.asyncio
    async def test_query_all_tasks(self):
        """Test querying all tasks without status filter."""
        mock_store = _make_mock_store()
        mock_store.list_tasks.return_value = [
            Task(id=1, name="Task 1", description="First", status=TaskStatus.PENDING, depends_on=[], result=None),
            Task(id=2, name="Task 2", description="Second", status=TaskStatus.RUNNING, depends_on=[], result=None),
        ]

        from koala.tools.task_tools import QueryTaskTool
        tool = QueryTaskTool(store=mock_store)

        result = await tool.ainvoke({})

        mock_store.list_tasks.assert_called_once_with(status=None)
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2

    @pytest.mark.asyncio
    async def test_query_tasks_by_status(self):
        """Test querying tasks filtered by status."""
        mock_store = _make_mock_store()
        mock_store.list_tasks.return_value = [
            Task(id=1, name="Task 1", description="First", status=TaskStatus.PENDING, depends_on=[], result=None),
        ]

        from koala.tools.task_tools import QueryTaskTool
        tool = QueryTaskTool(store=mock_store)

        result = await tool.ainvoke({"status": "PENDING"})

        mock_store.list_tasks.assert_called_once_with(status=TaskStatus.PENDING)
        assert len(result) == 1
        assert result[0]["status"] == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_query_empty_tasks(self):
        """Test querying when no tasks exist."""
        mock_store = _make_mock_store()
        mock_store.list_tasks.return_value = []

        from koala.tools.task_tools import QueryTaskTool
        tool = QueryTaskTool(store=mock_store)

        result = await tool.ainvoke({})

        assert result == []

    @pytest.mark.asyncio
    async def test_query_returns_list_of_dicts(self):
        """Test that query_tasks returns a list of dicts."""
        mock_store = _make_mock_store()
        mock_store.list_tasks.return_value = [
            Task(id=1, name="Task", description="Desc", status=TaskStatus.PENDING, depends_on=[], result=None),
        ]

        from koala.tools.task_tools import QueryTaskTool
        tool = QueryTaskTool(store=mock_store)

        result = await tool.ainvoke({})

        assert isinstance(result, list)
        assert isinstance(result[0], dict)


class TestAssignTaskTool:
    @pytest.mark.asyncio
    async def test_assign_task_registers_with_runner(self):
        """Test that assign_task calls register_background_task."""
        mock_store = _make_mock_store()
        mock_runner = _make_mock_runner()

        mock_store.get_task.return_value = Task(
            id=1,
            name="Task 1",
            description="Do something",
            status=TaskStatus.PENDING,
            depends_on=[],
            result=None
        )

        from koala.tools.task_tools import AssignTaskTool
        tool = AssignTaskTool(store=mock_store, runner=mock_runner)

        result = await tool.ainvoke({"task_id": 1})

        mock_store.get_task.assert_called_once_with(1)
        mock_runner.register_background_task.assert_called_once()
        call_args = mock_runner.register_background_task.call_args
        assert call_args[0][0] == 1  # task_id

    @pytest.mark.asyncio
    async def test_assign_nonexistent_task(self):
        """Test assigning a task that doesn't exist."""
        mock_store = _make_mock_store()
        mock_runner = _make_mock_runner()

        mock_store.get_task.return_value = None

        from koala.tools.task_tools import AssignTaskTool
        tool = AssignTaskTool(store=mock_store, runner=mock_runner)

        result = await tool.ainvoke({"task_id": 999})

        mock_runner.register_background_task.assert_not_called()
        assert "未找到" in result or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_assign_task_passes_coroutine(self):
        """Test that assign_task passes an async coroutine to the runner."""
        mock_store = _make_mock_store()
        mock_runner = _make_mock_runner()

        mock_store.get_task.return_value = Task(
            id=1,
            name="Task",
            description="Execute",
            status=TaskStatus.PENDING,
            depends_on=[],
            result=None
        )

        from koala.tools.task_tools import AssignTaskTool
        tool = AssignTaskTool(store=mock_store, runner=mock_runner)

        await tool.ainvoke({"task_id": 1})

        # Verify the second argument is awaitable
        call_args = mock_runner.register_background_task.call_args
        coro = call_args[0][1]
        import inspect
        assert inspect.iscoroutine(coro)

    @pytest.mark.asyncio
    async def test_assign_task_returns_success_message(self):
        """Test that assign_task returns a success message."""
        mock_store = _make_mock_store()
        mock_runner = _make_mock_runner()

        mock_store.get_task.return_value = Task(
            id=1,
            name="Task 1",
            description="Execute",
            status=TaskStatus.PENDING,
            depends_on=[],
            result=None
        )

        from koala.tools.task_tools import AssignTaskTool
        tool = AssignTaskTool(store=mock_store, runner=mock_runner)

        result = await tool.ainvoke({"task_id": 1})

        assert "已分配" in result or "assigned" in result.lower()
        assert "1" in result

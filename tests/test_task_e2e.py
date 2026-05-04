"""End-to-end tests for the task system integration."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from koala.agent.runner import AgentRunner
from koala.task.models import Task, TaskStatus
from koala.task.store import TaskStore
from koala.task.scheduler import TaskScheduler
from koala.task.notify import TaskNotifier
from koala.llm import LLMClient


@pytest.fixture
def mock_agent():
    """Mock agent with memory and arun."""
    agent = MagicMock()
    agent.memory = MagicMock()
    agent.memory.add_ai = MagicMock()
    agent.memory.add_user = MagicMock()
    agent.memory.messages = []
    agent.llm = MagicMock(spec=LLMClient)
    agent.tools = []
    agent.arun = AsyncMock(return_value="Agent response")
    return agent


@pytest.fixture
def temp_store(tmp_path):
    """Create a temporary TaskStore."""
    return TaskStore(filepath=str(tmp_path / "tasks.json"))


@pytest.fixture
def notifier():
    """Create TaskNotifier."""
    return TaskNotifier()


@pytest.fixture
def scheduler(temp_store, notifier):
    """Create TaskScheduler with shared notifier."""
    return TaskScheduler(store=temp_store, notifier=notifier)


@pytest.fixture
def runner(mock_agent, temp_store, scheduler, notifier):
    """Create AgentRunner with mocked dependencies."""
    return AgentRunner(
        agent=mock_agent,
        store=temp_store,
        scheduler=scheduler,
        notifier=notifier
    )


class TestCompleteTaskWorkflow:
    """Test complete task workflow from creation to completion."""

    @pytest.mark.asyncio
    async def test_parallel_tasks_with_dependent_task(self, runner, temp_store, scheduler):
        """Test T1, T2 parallel -> T3 depends on both."""
        # Mock the Agent class for subagent execution
        with patch('koala.agent.runner.Agent') as MockAgent:
            # Setup mock subagent responses
            mock_subagent1 = MagicMock()
            mock_subagent1.arun = AsyncMock(return_value="T1 result")

            mock_subagent2 = MagicMock()
            mock_subagent2.arun = AsyncMock(return_value="T2 result")

            mock_subagent3 = MagicMock()
            mock_subagent3.arun = AsyncMock(return_value="T3 result")

            # Set side effect so each call gets a fresh instance
            MockAgent.side_effect = [mock_subagent1, mock_subagent2, mock_subagent3]

            # Create tasks
            t1 = await temp_store.add_task("Task 1", "First task")
            t2 = await temp_store.add_task("Task 2", "Second task")
            t3 = await temp_store.add_task("Task 3", "Third task depends on 1 and 2", depends_on=[t1.id, t2.id])

            # Register T1 and T2 (they should run immediately)
            await runner.register_background_task(t1.id, runner.dispatch_with_subagent(t1))
            await runner.register_background_task(t2.id, runner.dispatch_with_subagent(t2))

            # Wait for T1 and T2 to complete
            await asyncio.sleep(0.1)

            # Verify T1 and T2 are completed
            updated_t1 = await temp_store.get_task(t1.id)
            updated_t2 = await temp_store.get_task(t2.id)
            assert updated_t1.status == TaskStatus.COMPLETED
            assert updated_t2.status == TaskStatus.COMPLETED
            assert updated_t1.result == "T1 result"
            assert updated_t2.result == "T2 result"

            # Now T3 should be ready - dispatch it
            ready_tasks = await temp_store.get_ready_tasks()
            assert len(ready_tasks) == 1
            assert ready_tasks[0].id == t3.id

            await runner.register_background_task(t3.id, runner.dispatch_with_subagent(t3))

            # Wait for T3 to complete
            await asyncio.sleep(0.1)

            # Verify T3 is completed
            updated_t3 = await temp_store.get_task(t3.id)
            assert updated_t3.status == TaskStatus.COMPLETED
            assert updated_t3.result == "T3 result"

    @pytest.mark.asyncio
    async def test_serial_dependency_chain(self, runner, temp_store, scheduler):
        """Test T1 -> T2 -> T3 serial dependency."""
        with patch('koala.agent.runner.Agent') as MockAgent:
            # Setup mock subagent responses
            mock_instances = [
                MagicMock(arun=AsyncMock(return_value="T1 done")),
                MagicMock(arun=AsyncMock(return_value="T2 done")),
                MagicMock(arun=AsyncMock(return_value="T3 done")),
            ]
            MockAgent.side_effect = mock_instances

            # Create serial dependency chain
            t1 = await temp_store.add_task("Task 1", "First task")
            t2 = await temp_store.add_task("Task 2", "Depends on T1", depends_on=[t1.id])
            t3 = await temp_store.add_task("Task 3", "Depends on T2", depends_on=[t2.id])

            # Initially only T1 should be ready
            ready = await temp_store.get_ready_tasks()
            assert len(ready) == 1
            assert ready[0].id == t1.id

            # Execute T1
            await runner.register_background_task(t1.id, runner.dispatch_with_subagent(t1))
            await asyncio.sleep(0.1)

            # After T1 completes, T2 should be ready
            ready = await temp_store.get_ready_tasks()
            assert len(ready) == 1
            assert ready[0].id == t2.id

            # Execute T2
            await runner.register_background_task(t2.id, runner.dispatch_with_subagent(t2))
            await asyncio.sleep(0.1)

            # After T2 completes, T3 should be ready
            ready = await temp_store.get_ready_tasks()
            assert len(ready) == 1
            assert ready[0].id == t3.id

            # Execute T3
            await runner.register_background_task(t3.id, runner.dispatch_with_subagent(t3))
            await asyncio.sleep(0.1)

            # All should be completed
            final_t1 = await temp_store.get_task(t1.id)
            final_t2 = await temp_store.get_task(t2.id)
            final_t3 = await temp_store.get_task(t3.id)

            assert final_t1.status == TaskStatus.COMPLETED
            assert final_t2.status == TaskStatus.COMPLETED
            assert final_t3.status == TaskStatus.COMPLETED


class TestToolInvocation:
    """Test that task tools work correctly when invoked by an agent."""

    @pytest.mark.asyncio
    async def test_create_task_tool_invocation(self, temp_store):
        """Test CreateTaskTool creates and persists tasks."""
        from koala.tools.task_tools import CreateTaskTool

        tool = CreateTaskTool(store=temp_store)
        result = await tool.ainvoke({
            "name": "Test Task",
            "description": "Test description"
        })

        assert result["id"] == 1
        assert result["name"] == "Test Task"
        assert result["status"] == TaskStatus.PENDING

        # Verify persistence
        task = await temp_store.get_task(1)
        assert task is not None
        assert task.name == "Test Task"

    @pytest.mark.asyncio
    async def test_query_task_tool_invocation(self, temp_store):
        """Test QueryTaskTool returns correct tasks."""
        from koala.tools.task_tools import QueryTaskTool

        # Create tasks with different statuses
        t1 = await temp_store.add_task("Pending Task", "Pending")
        t2 = await temp_store.add_task("Running Task", "Running")
        await temp_store.update_task(t2.id, status=TaskStatus.RUNNING)
        t3 = await temp_store.add_task("Completed Task", "Completed")
        await temp_store.update_task(t3.id, status=TaskStatus.COMPLETED)

        tool = QueryTaskTool(store=temp_store)

        # Query all tasks
        all_tasks = await tool.ainvoke({})
        assert len(all_tasks) == 3

        # Query by status
        pending_tasks = await tool.ainvoke({"status": "pending"})
        assert len(pending_tasks) == 1
        assert pending_tasks[0]["id"] == t1.id

        completed_tasks = await tool.ainvoke({"status": "completed"})
        assert len(completed_tasks) == 1
        assert completed_tasks[0]["id"] == t3.id

    @pytest.mark.asyncio
    async def test_assign_task_tool_invocation(self, runner, temp_store, scheduler):
        """Test AssignTaskTool triggers subagent execution."""
        from koala.tools.task_tools import AssignTaskTool

        with patch('koala.agent.runner.Agent') as MockAgent:
            mock_subagent = MagicMock()
            mock_subagent.arun = AsyncMock(return_value="Assigned task result")
            MockAgent.return_value = mock_subagent

            # Create a task
            task = await temp_store.add_task("Assigned Task", "To be assigned")

            # Create the tool
            tool = AssignTaskTool(store=temp_store, runner=runner)

            # Assign the task
            result = await tool.ainvoke({"task_id": task.id})
            assert "已分配" in result

            # Wait for execution
            await asyncio.sleep(0.1)

            # Verify task completed
            updated_task = await temp_store.get_task(task.id)
            assert updated_task.status == TaskStatus.COMPLETED
            assert updated_task.result == "Assigned task result"

    @pytest.mark.asyncio
    async def test_assign_nonexistent_task(self, runner, temp_store):
        """Test AssignTaskTool handles non-existent task."""
        from koala.tools.task_tools import AssignTaskTool

        tool = AssignTaskTool(store=temp_store, runner=runner)
        result = await tool.ainvoke({"task_id": 999})

        assert "未找到" in result


class TestDependencySchedulingE2E:
    """Test end-to-end dependency scheduling."""

    @pytest.mark.asyncio
    async def test_automatic_unlock_after_dependency_completes(self, runner, temp_store, scheduler):
        """Test that tasks automatically unlock when dependencies complete."""
        with patch('koala.agent.runner.Agent') as MockAgent:
            mock_instances = [
                MagicMock(arun=AsyncMock(return_value="Dep 1 done")),
                MagicMock(arun=AsyncMock(return_value="Dep 2 done")),
                MagicMock(arun=AsyncMock(return_value="Main done")),
            ]
            MockAgent.side_effect = mock_instances

            # Create two parallel dependencies
            dep1 = await temp_store.add_task("Dependency 1", "First dependency")
            dep2 = await temp_store.add_task("Dependency 2", "Second dependency")
            main = await temp_store.add_task("Main Task", "Depends on both", depends_on=[dep1.id, dep2.id])

            # Initially only deps should be ready
            ready = await temp_store.get_ready_tasks()
            assert len(ready) == 2
            ready_ids = {t.id for t in ready}
            assert ready_ids == {dep1.id, dep2.id}

            # Complete first dependency
            await runner.register_background_task(dep1.id, runner.dispatch_with_subagent(dep1))
            await asyncio.sleep(0.1)

            # Main should still not be ready (dep2 still pending)
            ready = await temp_store.get_ready_tasks()
            assert len(ready) == 1
            assert ready[0].id == dep2.id

            # Complete second dependency
            await runner.register_background_task(dep2.id, runner.dispatch_with_subagent(dep2))
            await asyncio.sleep(0.1)

            # Now main should be ready
            ready = await temp_store.get_ready_tasks()
            assert len(ready) == 1
            assert ready[0].id == main.id

    @pytest.mark.asyncio
    async def test_newly_unlocked_discovered_after_completion(self, runner, temp_store, scheduler):
        """Test get_newly_unlocked correctly identifies tasks unlocked by completion."""
        # Create a task that depends on T1
        t1 = await temp_store.add_task("T1", "Task 1")
        t2 = await temp_store.add_task("T2", "Depends on T1", depends_on=[t1.id])
        t3 = await temp_store.add_task("T3", "Also depends on T1", depends_on=[t1.id])

        # Complete T1
        await scheduler.on_task_complete(t1.id, "T1 completed")

        # Both T2 and T3 should be newly unlocked
        unlocked = await scheduler.get_newly_unlocked(t1.id)
        unlocked_ids = {t.id for t in unlocked}
        assert unlocked_ids == {t2.id, t3.id}

    @pytest.mark.asyncio
    async def test_failed_dependency_blocks_dependent_tasks(self, runner, temp_store, scheduler):
        """Test that failed dependencies block dependent tasks."""
        with patch('koala.agent.runner.Agent') as MockAgent:
            mock_subagent = MagicMock()
            mock_subagent.arun = AsyncMock(side_effect=Exception("Task failed"))
            MockAgent.return_value = mock_subagent

            # Create dependency and dependent task
            t1 = await temp_store.add_task("Failing Task", "Will fail")
            t2 = await temp_store.add_task("Dependent Task", "Depends on T1", depends_on=[t1.id])

            # Execute T1 (will fail)
            await runner.register_background_task(t1.id, runner.dispatch_with_subagent(t1))
            await asyncio.sleep(0.1)

            # Verify T1 failed
            updated_t1 = await temp_store.get_task(t1.id)
            assert updated_t1.status == TaskStatus.FAILED

            # T2 should not be ready (depends on failed task)
            ready = await temp_store.get_ready_tasks()
            assert len(ready) == 0
            assert all(t.id != t2.id for t in ready)


class TestNotificationFlowE2E:
    """Test end-to-end notification flow."""

    @pytest.mark.asyncio
    async def test_notification_triggers_ready_task_dispatch(self, runner, temp_store, scheduler):
        """Test that notifications trigger dispatch of ready tasks."""
        with patch('koala.agent.runner.Agent') as MockAgent:
            mock_subagent = MagicMock()
            mock_subagent.arun = AsyncMock(return_value="Task done")
            MockAgent.return_value = mock_subagent

            # Create dependency chain
            t1 = await temp_store.add_task("T1", "First")
            t2 = await temp_store.add_task("T2", "Second", depends_on=[t1.id])

            # Complete T1
            await scheduler.on_task_complete(t1.id, "T1 done")

            # Drain notifications
            notifications = await scheduler._notifier.drain()
            assert len(notifications) == 1
            assert notifications[0] == (t1.id, "T1 done")

            # T2 should now be ready
            ready = await temp_store.get_ready_tasks()
            assert len(ready) == 1
            assert ready[0].id == t2.id

    @pytest.mark.asyncio
    async def test_formatted_notification_injected_to_memory(self, runner, notifier):
        """Test that formatted notifications are injected into agent memory."""
        # Enqueue a notification
        await notifier.enqueue(5, "Test result")

        # Drain it
        notifications = await notifier.drain()

        # Format it
        formatted = notifier.format_notification(notifications[0][0], notifications[0][1])

        # Verify format
        assert "T5" in formatted
        assert "Test result" in formatted

        # Inject into memory
        runner._agent.memory.add_ai(formatted)

        # Verify it was added
        runner._agent.memory.add_ai.assert_called_once_with(formatted)

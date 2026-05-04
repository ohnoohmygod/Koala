import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from koala.agent.runner import AgentRunner
from koala.task.models import Task, TaskStatus
from koala.task.store import TaskStore
from koala.task.scheduler import TaskScheduler
from koala.task.notify import TaskNotifier


@pytest.fixture
def mock_agent():
    """Mock agent with memory and arun."""
    agent = MagicMock()
    agent.memory = MagicMock()
    agent.memory.add_ai = MagicMock()
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


@pytest.mark.asyncio
async def test_user_input_event_processing(runner, mock_agent):
    """Test that user input events are correctly processed."""
    # Run the loop in background
    run_task = asyncio.create_task(runner.run())

    # Simulate user input
    await runner.submit_user_input("Hello")

    # Give it time to process
    await asyncio.sleep(0.1)

    # Stop the loop and cancel the task
    runner._running = False
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass

    # Verify agent.arun was called with "Hello"
    mock_agent.arun.assert_called()
    # Check that "Hello" was one of the calls
    calls = [call[0][0] for call in mock_agent.arun.call_args_list]
    assert "Hello" in calls


@pytest.mark.asyncio
async def test_background_task_completion_notification(runner, mock_agent, scheduler):
    """Test that background task completion events are correctly processed."""
    # Run the loop in background
    run_task = asyncio.create_task(runner.run())

    # Create a completed task notification
    await scheduler._notifier.enqueue(1, "Task completed")

    # Give it time to process
    await asyncio.sleep(0.1)

    # Stop the loop and cancel the task
    runner._running = False
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass

    # Verify notification was added to agent memory
    mock_agent.memory.add_ai.assert_called()
    call_args = mock_agent.memory.add_ai.call_args[0][0]
    assert "T1" in call_args
    assert "Task completed" in call_args


@pytest.mark.asyncio
async def test_notification_drain_and_memory_injection(runner, mock_agent, notifier):
    """Test that notifications are drained and injected into agent memory."""
    # Run the loop in background
    run_task = asyncio.create_task(runner.run())

    # Enqueue multiple notifications
    await notifier.enqueue(1, "First result")
    await notifier.enqueue(2, "Second result")

    # Give it time to process
    await asyncio.sleep(0.1)

    # Stop the loop
    runner._running = False
    await runner.submit_user_input("")  # Wake up the loop
    try:
        await asyncio.wait_for(run_task, timeout=1.0)
    except asyncio.TimeoutError:
        run_task.cancel()

    # Both notifications should be in memory
    assert mock_agent.memory.add_ai.call_count == 2


@pytest.mark.asyncio
async def test_register_background_task(runner, temp_store, scheduler):
    """Test that register_background_task registers and calls scheduler on completion."""
    # Create a task
    task = await temp_store.add_task("Test task", "Test description")

    # Mock the scheduler methods
    scheduler.on_task_complete = AsyncMock()

    # Register a background task
    async def mock_coro():
        return "Background result"

    await runner.register_background_task(task.id, mock_coro())

    # Give it a moment to complete
    await asyncio.sleep(0.1)

    # Verify scheduler.on_task_complete was called
    scheduler.on_task_complete.assert_called_once_with(task.id, "Background result")


@pytest.mark.asyncio
async def test_dual_source_competition(runner, mock_agent, notifier):
    """Test that user input and background task completion are handled correctly when both arrive."""
    # Run the loop in background
    run_task = asyncio.create_task(runner.run())

    # Enqueue a background notification
    await notifier.enqueue(1, "Background task result")

    # Submit user input
    await runner.submit_user_input("User message")

    # Give it time to process both
    await asyncio.sleep(0.1)

    # Stop the loop and cancel the task
    runner._running = False
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass

    # Both should be processed
    # At minimum, agent.arun should be called for user input
    assert mock_agent.arun.called or mock_agent.memory.add_ai.called


@pytest.mark.asyncio
async def test_dispatch_with_subagent(runner, temp_store, scheduler):
    """Test that dispatch_with_subagent creates a subagent and handles completion."""
    # Mock the scheduler.on_task_complete
    scheduler.on_task_complete = AsyncMock()

    # Create a task
    task = await temp_store.add_task("Test task", "Test description")

    # Mock the Agent class to avoid actual LLM calls
    with patch('koala.agent.runner.Agent') as MockAgent:
        mock_agent_instance = MagicMock()
        mock_agent_instance.arun = AsyncMock(return_value="Subagent result")
        MockAgent.return_value = mock_agent_instance

        # Dispatch the task
        result = await runner.dispatch_with_subagent(task)

        # Verify the result
        assert result == "Subagent result"


@pytest.mark.asyncio
async def test_empty_iteration(runner, mock_agent):
    """Test that the event loop waits when there are no events."""
    # Run the loop in background
    run_task = asyncio.create_task(runner.run())

    # Give it a moment - it should be waiting
    await asyncio.sleep(0.05)

    # Stop the loop and cancel the task
    runner._running = False
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass

    # Should not fail, agent should not be called
    assert not mock_agent.arun.called


@pytest.mark.asyncio
async def test_running_flag_controls_loop(runner):
    """Test that the running flag controls the event loop."""
    # Start with running=True
    runner._running = True

    # Run should continue until running=False
    async def stop_after_delay():
        await asyncio.sleep(0.1)
        runner.stop()
        # Wake up the event loop so it can check the flag
        await runner.submit_user_input("__stop__")

    # Run both tasks
    run_task = asyncio.create_task(runner.run())
    stop_task = asyncio.create_task(stop_after_delay())

    # Wait for both to complete
    await asyncio.gather(run_task, stop_task)

    # Test passes if we get here without hanging
    assert True

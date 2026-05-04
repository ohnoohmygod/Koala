import asyncio
import logging
from typing import Callable, Awaitable

from koala.agent.agent import Agent
from koala.task.store import TaskStore
from koala.task.scheduler import TaskScheduler
from koala.task.notify import TaskNotifier
from koala.task.models import Task
from koala.llm import LLMClient


logger = logging.getLogger(__name__)


class AgentRunner:
    """Event loop runner for agent with task scheduling support."""

    def __init__(
        self,
        agent: Agent,
        store: TaskStore,
        scheduler: TaskScheduler,
        notifier: TaskNotifier
    ):
        """Initialize the runner with dependencies.

        Args:
            agent: Agent instance for processing user input
            store: TaskStore for task persistence
            scheduler: TaskScheduler for task execution
            notifier: TaskNotifier for completion notifications
        """
        self._agent = agent
        self._store = store
        self._scheduler = scheduler
        self._notifier = notifier

        # Queue for user input
        self._user_queue: asyncio.Queue[str] = asyncio.Queue()

        # Registry for background tasks (task_id -> asyncio.Task)
        self._background_tasks: dict[int, asyncio.Task] = {}

        # Control flag for event loop
        self._running = True

    def stop(self) -> None:
        """Stop the event loop."""
        self._running = False

    async def submit_user_input(self, text: str) -> None:
        """Submit user input to the queue.

        Args:
            text: User input text
        """
        await self._user_queue.put(text)

    async def run(self) -> None:
        """Main event loop.

        Continuously processes:
        1. Drains notification queue and injects into agent memory
        2. Waits for user input OR background task completion
        3. Processes completed events
        4. Dispatches ready tasks when notifications arrive
        """
        while self._running:
            # 1. Drain notification queue and inject into agent memory
            notifications = await self._notifier.drain()
            for task_id, result in notifications:
                msg = self._notifier.format_notification(task_id, result)
                self._agent.memory.add_ai(msg)

            if not self._running:
                break

            # 2. Wait for events: user input OR background task completion
            user_future = asyncio.create_task(self._user_queue.get())
            bg_future = asyncio.create_task(self._notifier.wait())

            done, pending = await asyncio.wait(
                [user_future, bg_future],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending futures
            for fut in pending:
                fut.cancel()

            if not self._running:
                break

            # 3. Process completed events
            for task in done:
                try:
                    result = task.result()

                    # Check if result is user input (string) or background notification (tuple)
                    if isinstance(result, str):
                        # User input - process with agent
                        response = await self._agent.arun(result)
                        logger.debug(f"Agent response: {response}")
                    elif isinstance(result, tuple) and len(result) == 2:
                        # Background notification - trigger scheduler to check new tasks
                        await self._scheduler.dispatch_ready_tasks(self.dispatch_with_subagent)

                except Exception as e:
                    logger.error(f"Error processing event: {e}")

    async def register_background_task(self, task_id: int, coro: Awaitable[str]) -> None:
        """Register a background task that auto-completes on finish.

        Args:
            task_id: Task ID for tracking
            coro: Coroutine to execute
        """
        async def wrapped():
            try:
                result = await coro
                await self._scheduler.on_task_complete(task_id, result)
            except Exception as e:
                await self._scheduler.on_task_failed(task_id, str(e))

        bg_task = asyncio.create_task(wrapped())
        self._background_tasks[task_id] = bg_task

    async def dispatch_with_subagent(self, task: Task) -> str:
        """Executor function that creates a subagent to execute a task.

        Args:
            task: Task to execute

        Returns:
            Result string from task execution
        """
        # Create a subagent with minimal context
        # We use the same LLM but with a focused system prompt
        sub_prompt = "你是一个子任务执行者。完成指定任务后，用简洁的文字总结结果。"
        subagent = Agent(
            llm=self._agent.llm,
            tools=self._agent.tools,
            system_prompt=sub_prompt
        )

        try:
            result = await subagent.arun(task.description)
            return result
        except Exception as e:
            logger.error(f"Subagent error for task {task.id}: {e}")
            raise

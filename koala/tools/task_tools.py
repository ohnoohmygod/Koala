"""LangChain tools for agent to interact with the task system."""
import asyncio
from typing import Optional, Type, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from koala.task.models import Task, TaskStatus
from koala.task.store import TaskStore
from koala.agent.runner import AgentRunner


class CreateTaskInput(BaseModel):
    """Input schema for CreateTaskTool."""
    name: str = Field(description="任务名称")
    description: str = Field(description="任务详细描述")
    depends_on: list[int] = Field(default=[], description="依赖的任务ID列表")


class CreateTaskTool(BaseTool):
    """Tool for creating new tasks in the task store."""
    name: str = "create_task"
    description: str = "创建一个新任务。参数：name(任务名称), description(任务描述), depends_on(可选的依赖任务ID列表)"
    args_schema: Type[BaseModel] = CreateTaskInput
    store: Any  # TaskStore or mock for testing

    def _run(self, name: str, description: str, depends_on: list[int] | None = None) -> dict:
        raise NotImplementedError("请使用异步版本 ainvoke")

    async def _arun(self, name: str, description: str, depends_on: list[int] | None = None) -> dict:
        """Create a new task in the store.

        Args:
            name: Task name
            description: Task description
            depends_on: Optional list of task IDs this task depends on

        Returns:
            Dictionary representation of the created task
        """
        task = await self.store.add_task(
            name=name,
            description=description,
            depends_on=depends_on or []
        )
        return task.to_dict()


class QueryTaskInput(BaseModel):
    """Input schema for QueryTaskTool."""
    status: Optional[str] = Field(default=None, description="可选的状态过滤：pending/running/completed/failed")


class QueryTaskTool(BaseTool):
    """Tool for querying tasks from the task store."""
    name: str = "query_tasks"
    description: str = "查询任务列表。参数：status(可选，按状态过滤：pending/running/completed/failed)"
    args_schema: Type[BaseModel] = QueryTaskInput
    store: Any  # TaskStore or mock for testing

    def _run(self, status: Optional[str] = None) -> list:
        raise NotImplementedError("请使用异步版本 ainvoke")

    async def _arun(self, status: Optional[str] = None) -> list:
        """Query tasks from the store.

        Args:
            status: Optional status filter (pending/running/completed/failed)

        Returns:
            List of task dictionaries
        """
        status_enum = None
        if status:
            status_map = {
                "pending": TaskStatus.PENDING,
                "running": TaskStatus.RUNNING,
                "completed": TaskStatus.COMPLETED,
                "failed": TaskStatus.FAILED,
            }
            status_enum = status_map.get(status.lower())
            if status_enum is None:
                return []

        tasks = await self.store.list_tasks(status=status_enum)
        return [task.to_dict() for task in tasks]


class AssignTaskInput(BaseModel):
    """Input schema for AssignTaskTool."""
    task_id: int = Field(description="要分配的任务ID")


class AssignTaskTool(BaseTool):
    """Tool for assigning a task to be executed by a subagent."""
    name: str = "assign_task"
    description: str = "将任务分配给子代理执行。参数：task_id(要分配的任务ID)"
    args_schema: Type[BaseModel] = AssignTaskInput
    store: Any  # TaskStore or mock for testing
    runner: Any  # AgentRunner or mock for testing

    def _run(self, task_id: int) -> str:
        raise NotImplementedError("请使用异步版本 ainvoke")

    async def _arun(self, task_id: int) -> str:
        """Assign a task to be executed by a subagent.

        Args:
            task_id: The ID of the task to assign

        Returns:
            Status message
        """
        task = await self.store.get_task(task_id)
        if task is None:
            return f"任务 {task_id} 未找到"

        # Create a coroutine that will execute the task
        async def execute_task():
            return await self.runner.dispatch_with_subagent(task)

        # Register the background task with the runner
        await self.runner.register_background_task(task_id, execute_task())

        return f"任务 {task_id} 已分配给子代理执行"

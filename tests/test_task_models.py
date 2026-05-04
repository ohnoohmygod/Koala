import pytest
from koala.task.models import Task, TaskStatus


def test_task_creation_with_defaults():
    task = Task(id=1, name="测试任务", description="这是一个测试任务")
    assert task.id == 1
    assert task.name == "测试任务"
    assert task.description == "这是一个测试任务"
    assert task.status == TaskStatus.PENDING
    assert task.depends_on == []
    assert task.result is None


def test_task_creation_with_all_fields():
    task = Task(
        id=2,
        name="完整任务",
        description="包含所有字段",
        status=TaskStatus.RUNNING,
        depends_on=[1, 3],
        result="进行中"
    )
    assert task.id == 2
    assert task.name == "完整任务"
    assert task.description == "包含所有字段"
    assert task.status == TaskStatus.RUNNING
    assert task.depends_on == [1, 3]
    assert task.result == "进行中"


def test_task_status_transitions():
    task = Task(id=1, name="状态测试", description="测试状态转换")

    assert task.status == TaskStatus.PENDING

    task.status = TaskStatus.RUNNING
    assert task.status == TaskStatus.RUNNING

    task.status = TaskStatus.COMPLETED
    assert task.status == TaskStatus.COMPLETED

    task.status = TaskStatus.FAILED
    assert task.status == TaskStatus.FAILED


def test_task_to_dict():
    task = Task(
        id=5,
        name="序列化测试",
        description="测试转换为字典",
        status=TaskStatus.COMPLETED,
        depends_on=[1, 2],
        result="成功完成"
    )

    result = task.to_dict()
    assert result == {
        "id": 5,
        "name": "序列化测试",
        "description": "测试转换为字典",
        "status": TaskStatus.COMPLETED,
        "depends_on": [1, 2],
        "result": "成功完成"
    }


def test_task_from_dict():
    data = {
        "id": 10,
        "name": "反序列化测试",
        "description": "测试从字典创建",
        "status": TaskStatus.FAILED,
        "depends_on": [5, 6],
        "result": "执行失败"
    }

    task = Task.from_dict(data)
    assert task.id == 10
    assert task.name == "反序列化测试"
    assert task.description == "测试从字典创建"
    assert task.status == TaskStatus.FAILED
    assert task.depends_on == [5, 6]
    assert task.result == "执行失败"


def test_task_serialization_roundtrip():
    original = Task(
        id=100,
        name="往返测试",
        description="测试序列化反序列化往返",
        status=TaskStatus.RUNNING,
        depends_on=[1, 2, 3],
        result="正在处理"
    )

    # 序列化
    data = original.to_dict()

    # 反序列化
    restored = Task.from_dict(data)

    # 验证所有字段
    assert restored.id == original.id
    assert restored.name == original.name
    assert restored.description == original.description
    assert restored.status == original.status
    assert restored.depends_on == original.depends_on
    assert restored.result == original.result


def test_task_empty_description():
    task = Task(id=1, name="空描述", description="")
    assert task.description == ""
    assert task.name == "空描述"


def test_task_no_dependencies():
    task = Task(id=1, name="无依赖", description="没有依赖的任务")
    assert task.depends_on == []

    # 也可以显式传入空列表
    task2 = Task(id=2, name="显式空依赖", description="显式传入空列表", depends_on=[])
    assert task2.depends_on == []


def test_task_status_enum_values():
    # 验证 enum 值等于字符串（因为继承自 str）
    assert TaskStatus.PENDING == "PENDING"
    assert TaskStatus.RUNNING == "RUNNING"
    assert TaskStatus.COMPLETED == "COMPLETED"
    assert TaskStatus.FAILED == "FAILED"

    # 验证可以作为字符串使用
    status = TaskStatus.PENDING
    assert status == "PENDING"
    # 在字符串上下文中使用
    message = f"任务状态: {status}"
    assert "PENDING" in message


def test_task_multiple_dependencies():
    task = Task(
        id=10,
        name="多依赖任务",
        description="依赖多个前置任务",
        depends_on=[1, 2, 3, 4, 5]
    )
    assert len(task.depends_on) == 5
    assert 1 in task.depends_on
    assert 5 in task.depends_on


def test_task_result_can_be_none():
    task = Task(id=1, name="无结果", description="还没有结果的任务")
    assert task.result is None

    task.result = "已完成"
    assert task.result == "已完成"

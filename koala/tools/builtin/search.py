from langchain_core.tools import tool


@tool
def search(query: str) -> str:
    """搜索互联网信息。query: 搜索关键词。"""
    # TODO: 接入真实搜索 API
    return f"搜索结果：{query} 的相关信息..."


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。expression: 数学表达式，如 '2 + 3 * 4'。"""
    try:
        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

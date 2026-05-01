import subprocess
from pathlib import Path
from langchain_core.tools import tool


@tool
def read_file(path: str) -> str:
    """读取文件内容。path: 文件的绝对或相对路径。"""
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"错误: 文件不存在 {path}"
    except Exception as e:
        return f"错误: {e}"


@tool
def glob_search(pattern: str, directory: str = ".") -> str:
    """按文件名模式查找文件。pattern: glob 模式，如 '**/*.py'。directory: 搜索目录，默认当前目录。"""
    matches = sorted(Path(directory).glob(pattern))
    if not matches:
        return "未找到匹配文件"
    return "\n".join(str(m) for m in matches)


@tool
def grep_search(pattern: str, directory: str = ".") -> str:
    """在文件中搜索文本。pattern: 要搜索的文本。directory: 搜索目录，默认当前目录。"""
    results = []
    for path in Path(directory).rglob("*"):
        if path.is_file():
            try:
                for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if pattern in line:
                        results.append(f"{path}:{i}: {line.strip()}")
            except (UnicodeDecodeError, PermissionError):
                continue
    if not results:
        return "未找到匹配内容"
    return "\n".join(results)


@tool
def bash(command: str) -> str:
    """执行 bash 命令并返回输出。command: 要执行的命令。"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if result.returncode != 0:
            output += f"\n错误: {result.stderr.strip()}"
        return output
    except subprocess.TimeoutExpired:
        return "错误: 命令执行超时"
    except Exception as e:
        return f"错误: {e}"

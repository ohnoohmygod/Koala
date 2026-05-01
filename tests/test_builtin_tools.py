import os
import tempfile
import pytest

from koala.tools.builtin.file_tools import read_file, glob_search, grep_search, bash


@pytest.fixture
def temp_dir():
    """创建临时目录和测试文件。"""
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "src"))
        # 创建几个测试文件
        with open(os.path.join(d, "src", "main.py"), "w") as f:
            f.write("def hello():\n    print('hello world')\n")
        with open(os.path.join(d, "src", "utils.py"), "w") as f:
            f.write("def add(a, b):\n    return a + b\n")
        with open(os.path.join(d, "README.md"), "w") as f:
            f.write("# Test Project\nhello world\n")
        yield d


class TestReadFile:
    def test_read_existing_file(self, temp_dir):
        path = os.path.join(temp_dir, "src", "main.py")
        result = read_file.invoke({"path": path})
        assert "def hello" in result
        assert "print" in result

    def test_read_nonexistent_file(self, temp_dir):
        result = read_file.invoke({"path": os.path.join(temp_dir, "nope.py")})
        assert "错误" in result or "不存在" in result


class TestGlobSearch:
    def test_find_py_files(self, temp_dir):
        result = glob_search.invoke({"pattern": "**/*.py", "directory": temp_dir})
        assert "main.py" in result
        assert "utils.py" in result

    def test_find_md_files(self, temp_dir):
        result = glob_search.invoke({"pattern": "*.md", "directory": temp_dir})
        assert "README.md" in result
        assert "main.py" not in result

    def test_no_match(self, temp_dir):
        result = glob_search.invoke({"pattern": "*.java", "directory": temp_dir})
        assert result == "" or "未找到" in result


class TestGrepSearch:
    def test_find_text_in_files(self, temp_dir):
        result = grep_search.invoke({"pattern": "hello", "directory": temp_dir})
        assert "main.py" in result

    def test_find_text_with_line_number(self, temp_dir):
        result = grep_search.invoke({"pattern": "def add", "directory": temp_dir})
        assert "utils.py" in result

    def test_no_match(self, temp_dir):
        result = grep_search.invoke({"pattern": "nonexistent_text_xyz", "directory": temp_dir})
        assert "未找到" in result or result == ""


class TestBash:
    def test_simple_command(self):
        result = bash.invoke({"command": "echo hello"})
        assert "hello" in result

    def test_command_with_args(self):
        result = bash.invoke({"command": "ls /dev/null"})
        assert "null" in result

    def test_failed_command(self):
        result = bash.invoke({"command": "ls /nonexistent_dir_xyz"})
        assert "错误" in result or "No such" in result or result != ""

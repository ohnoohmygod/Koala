# Koala

多 Agent AI 助手框架，基于 LangChain 构建，支持工具调用和子 Agent 异步并发。

## 特性

- **多 LLM 提供商** — 支持 GLM (智谱)、DeepSeek，运行时动态切换
- **Tool Calling** — Agent 自动选择并执行工具，支持同步和异步
- **内置工具** — 搜索、计算器、文件读取、文件查找、内容搜索、Bash 命令执行
- **SubAgent 并发** — 多个子任务通过 `asyncio.gather` 异步并发执行
- **上下文继承** — SubAgent 可选择继承父 Agent 的对话历史

## 快速开始

```bash
pip install -e ".[dev]"
cp .env.example .env   # 填入你的 API Key
python main.py
```

## 项目结构

```
koala/
├── llm.py                  # LLM 客户端，多提供商动态切换
├── agent/
│   └── agent.py            # Agent 核心循环（tool calling + 记忆管理）
├── tools/
│   ├── builtin/
│   │   ├── search.py       # 搜索、计算器
│   │   └── file_tools.py   # 文件读取、glob 查找、grep 搜索、Bash 执行
│   └── subagent.py         # SubAgent 异步并发工具
└── context/
    └── memory.py           # 短期记忆（消息管理 + token 估算）
```

## 开发进度

### 已完成

- [x] 项目骨架（pyproject.toml、目录结构）
- [x] LLMClient — LangChain + OpenAI 兼容 API，支持 GLM / DeepSeek 动态切换
- [x] Agent 基类 — tool calling 循环，同步 + 异步
- [x] 内置工具 — search、calculator、read_file、glob_search、grep_search、bash
- [x] SubAgentTool — 多子任务 `asyncio.gather` 并发执行，支持上下文继承
- [x] ShortTermMemory — 消息管理，token 估算

### 进行中

- [ ] 配置管理 — 统一管理 provider、model、工具等配置
- [ ] 上下文压缩 — 摘要 / 滑动窗口，避免 token 溢出

### 计划中

- [ ] 长期记忆 — 跨对话持久化
- [ ] MCP 协议支持 — 接入外部工具生态
- [ ] Web UI — 对话界面
- [ ] 多 Agent 协作框架 — Agent 间通信、任务分解与分配

## 开发

```bash
pytest tests/ -v
```

## 项目导航

- [ROADMAP.md](./ROADMAP.md) — 详细架构设计和开发计划

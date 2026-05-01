# Koala Roadmap

多 Agent AI 助手项目。

## 架构

```
koala/
├── llm.py              # LLM 客户端，支持多提供商动态切换
├── agent/
│   ├── base.py         # Agent 基类（单 agent 对话循环）
│   └── runner.py       # 多 agent 调度（SubAgent 异步并发）
├── tools/
│   ├── registry.py     # 工具注册中心
│   └── builtin/        # 内置工具
├── context/
│   ├── memory.py       # 短期记忆 ✅ / 长期记忆 🔜 / 压缩 🔜
│   └── compressor.py   # 上下文压缩（未实现）
└── config.py           # 配置管理（未实现）
```

## 进度

### Phase 1: 基础设施 ✅
- [x] 项目初始化（pyproject.toml, 目录结构）
- [x] LLMClient（LangChain + OpenAI 兼容 API，支持 GLM/DeepSeek 动态切换）
- [x] ToolRegistry（注册/发现 LangChain BaseTool）
- [x] 内置工具（search, calculator 示例）
- [x] Agent 基类（tool calling 循环，同步+异步）
- [x] ShortTermMemory（消息管理，token 估算）

### Phase 2: SubAgent 异步并发 🔜
- [ ] SubAgent 定义（独立 system_prompt、工具集）
- [ ] AgentRunner（asyncio.gather 并发调度多个 SubAgent）
- [ ] 结果聚合与回传

### Phase 3: 上下文增强 🔜
- [ ] 长期记忆（跨对话持久化）
- [ ] 上下文压缩（摘要/滑动窗口）

### Phase 4: Team 模式（远期）
- [ ] 多 Agent 协作框架
- [ ] Agent 间通信
- [ ] 任务分解与分配

## 开发规范

- TDD：先写测试，再写实现
- 所有测试在 `tests/` 目录，文件对应 `koala/` 下的模块

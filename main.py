import asyncio
from koala.llm import LLMClient
from koala.agent.agent import Agent
from koala.tools.builtin.search import search, calculator
from koala.tools.builtin.file_tools import read_file, glob_search, grep_search, bash
from koala.tools.subagent import SubAgentTool


async def main():
    llm = LLMClient("deepseek")
    sub_tool = SubAgentTool(llm=llm, tools=[search, calculator, read_file, glob_search, grep_search, bash])
    agent = Agent(
        llm=llm,
        tools=[search, calculator, read_file, glob_search, grep_search, bash, sub_tool],
    )

    print("Koala Agent 已启动，输入问题开始对话，输入 exit 退出\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("再见！")
            break

        result = await agent.arun(user_input)
        print(f"Agent: {result}\n")


if __name__ == "__main__":
    asyncio.run(main())

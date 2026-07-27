import asyncio
import sys
import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_stock_agent(user_query: str):
    # Pass sys.executable to ensure MCP uses your venv Python interpreter
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 1. Fetch MCP tools
            tools_response = await session.list_tools()
            available_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                }
                for tool in tools_response.tools
            ]

            messages = [
                {"role": "system", "content": "You are an expert quantitative financial analyst AI. Use tools to fetch stock data and predictions before answering."},
                {"role": "user", "content": user_query}
            ]

            print(f"🤖 User Query: {user_query}\n")

            # 2. Query Ollama with tools
            response = ollama.chat(
                model="qwen2.5:7b",
                messages=messages,
                tools=available_tools
            )

            # 3. Handle tool calls
            if response.get("message", {}).get("tool_calls"):
                for tool_call in response["message"]["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]

                    print(f"🛠️ [MCP Executing]: {tool_name}({tool_args})...")
                    
                    result = await session.call_tool(tool_name, tool_args)
                    tool_output = result.content[0].text

                    messages.append(response["message"])
                    messages.append({
                        "role": "tool",
                        "content": tool_output
                    })

                # 4. Synthesize final answer & unload model from RAM when finished
                final_response = ollama.chat(
                    model="qwen2.5:7b",
                    messages=messages,
                    keep_alive=0  # Frees up RAM immediately after completion
                )
                print("\n📈 Final Financial Report:")
                print(final_response["message"]["content"])
            else:
                print(response["message"]["content"])

if __name__ == "__main__":
    prompt = "Can you check NVDA stock metrics and run a 5-day ML trend prediction?"
    asyncio.run(run_stock_agent(prompt))
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_servers/battery_soh_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()

            for tool in tools_result.tools:
                print("Tool:", tool.name)
                print("Input schema:", tool.inputSchema)
                print("-" * 50)


asyncio.run(main())
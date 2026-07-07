#!/usr/bin/env python
"""
MCP example: get_providers tool

Lists all LLM providers and shows which ones have API keys configured.
No API key required for this tool itself.

Usage:
    pip install "syda[mcp]"
    python examples/mcp/test_get_providers.py
"""

import asyncio
import json
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(command="syda-mcp")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("=" * 60)
            print("get_providers — checking configured LLM providers")
            print("=" * 60)

            result = await session.call_tool("get_providers", {})
            data = json.loads(result.content[0].text)

            print(f"\n{data['message']}\n")

            for p in data["providers"]:
                status = "✓ configured" if p["configured"] else "✗ not set"
                print(f"  {p['provider']:<20} {status}")
                print(f"    env var  : {p['env_var']}")
                print(f"    model    : {p['recommended_model']}")
                print(f"    notes    : {p['notes']}")
                print()

            # Show which would be auto-selected
            configured = data["configured"]
            if configured:
                print(f"Auto-detection order: {' → '.join(configured)}")
                print(f"Default provider (no 'provider' param): {configured[0]}")
            else:
                print("No providers configured — set at least one API key.")
                print("\nQuick setup:")
                print("  export ANTHROPIC_API_KEY=sk-ant-...")
                print("  export OPENAI_API_KEY=sk-proj-...")
                print("  export GROK_API_KEY=xai-...")


if __name__ == "__main__":
    asyncio.run(main())

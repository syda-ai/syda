#!/usr/bin/env python
"""
Example: calling Syda MCP tools programmatically via the MCP Python SDK.

This demonstrates how to call the same tools that Claude Desktop / Cursor
use, but from Python code — useful for testing or automation.

For normal use, just configure Claude Desktop / Cursor and ask naturally:
  "Generate a fintech dataset with 500 customers and 2,000 transactions"

Requirements:
    pip install "syda[mcp]" mcp
    ANTHROPIC_API_KEY (or any other provider key) in environment
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(
        command="syda-mcp",
        args=[],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ── List available tools ──────────────────────────────────────────
            tools = await session.list_tools()
            print("Available Syda MCP tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description[:60]}...")
            print()

            # ── Check configured providers ────────────────────────────────────
            result = await session.call_tool("get_providers", {})
            providers_data = json.loads(result.content[0].text)
            print(f"Configured providers: {providers_data['configured']}")
            print()

            # ── Validate a schema first ───────────────────────────────────────
            schema = {
                "customers": {
                    "customer_id": {"type": "integer", "primary_key": True},
                    "email":       {"type": "email",   "unique": True},
                    "name":        {"type": "string"},
                    "country":     {"type": "string",
                                   "enum": ["US", "UK", "DE", "FR", "IN"]},
                    "plan":        {"type": "string",
                                   "enum": ["free", "pro", "enterprise"],
                                   "probabilities": [0.6, 0.3, 0.1]},
                },
                "transactions": {
                    "tx_id":       {"type": "integer", "primary_key": True},
                    "customer_id": {
                        "type": "foreign_key",
                        "references": {"schema": "customers", "field": "customer_id"},
                    },
                    "amount":      {"type": "float", "min": 1.0, "max": 5000.0},
                    "status":      {"type": "string",
                                   "enum": ["pending", "completed", "failed"],
                                   "probabilities": [0.1, 0.8, 0.1]},
                    "tx_date":     {"type": "date"},
                },
            }

            val = await session.call_tool("validate_schema", {"schema": schema})
            val_data = json.loads(val.content[0].text)
            print(f"Schema validation: {'OK' if val_data['ok'] else 'FAILED'}")
            if val_data["warnings"]:
                for w in val_data["warnings"]:
                    print(f"  Warning: {w}")
            print(f"  Tables: {list(val_data['tables'].keys())}")
            print(f"  Relationships: {val_data['relationships']}")
            print()

            if not val_data["ok"]:
                print("Schema invalid — fix errors before generating.")
                return

            # ── Generate data ─────────────────────────────────────────────────
            print("Generating fintech dataset...")
            gen = await session.call_tool(
                "generate_from_schema",
                {
                    "schema": schema,
                    "sample_sizes": {"customers": 20, "transactions": 50},
                    "prompts": {
                        "customers": "Mix of US, UK, and European fintech customers",
                        "transactions": "Realistic payment transactions, mostly completed",
                    },
                    "preview_rows": 3,
                },
            )
            gen_data = json.loads(gen.content[0].text)

            if not gen_data["ok"]:
                print(f"Generation failed: {gen_data['message']}")
                return

            print(f"✓ {gen_data['message']}")
            print()

            # Row counts
            for table, count in gen_data["row_counts"].items():
                print(f"  {table}: {count:,} rows")

            # Cost report
            report = gen_data.get("report", {})
            if report:
                print(f"\n  LLM calls : {report.get('total_llm_calls', '?')}")
                print(f"  Cost      : ${report.get('estimated_cost_usd', 0):.4f}")
                print(f"  Time      : {report.get('total_time_s', '?')}s")
                print(f"  Model     : {report.get('model', '?')}")

            # Sample rows
            print("\n── Sample customers ────────────────────────────────────────")
            for row in gen_data["tables"].get("customers", []):
                print(f"  {row}")

            print("\n── Sample transactions ─────────────────────────────────────")
            for row in gen_data["tables"].get("transactions", []):
                print(f"  {row}")


if __name__ == "__main__":
    asyncio.run(main())

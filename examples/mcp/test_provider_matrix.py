#!/usr/bin/env python
"""
MCP example: provider matrix — generate_from_schema forced across every
supported provider, one at a time, verifying real FK referential integrity
for each.

Unlike test_generate_schemas.py (which lets generate_from_schema auto-detect
the provider), this forces `provider=` explicitly per run — the only way to
isolate a specific provider's code path when multiple API keys are
configured at once (auto-detection always picks the first one found, in a
fixed priority order).

Any provider without a configured API key is skipped rather than failing
the whole run. openai_compatible has no API key to gate on — it's attempted
unconditionally against a local Ollama endpoint by default and reported as
a failure (not a skip) if unreachable.

Usage:
    pip install "syda[mcp]"
    # set whichever provider keys you have in .env; for openai_compatible,
    # run Ollama locally (https://ollama.com) and `ollama pull <model>`,
    # or point OPENAI_COMPATIBLE_BASE_URL at another OpenAI-compatible API.
    python examples/mcp/test_provider_matrix.py
"""

import asyncio
import csv
import json
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SCHEMA = {
    "customers": {
        "customer_id": {"type": "integer", "primary_key": True},
        "name":        {"type": "string"},
        "email":       {"type": "email", "unique": True},
    },
    "orders": {
        "order_id":    {"type": "integer", "primary_key": True},
        "customer_id": {"type": "foreign_key",
                         "references": {"schema": "customers", "field": "customer_id"}},
        "amount":      {"type": "float", "min": 5.0, "max": 500.0},
        "status":      {"type": "string", "enum": ["pending", "shipped", "delivered"]},
    },
}

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "provider_matrix")

PROVIDERS = [
    {
        "label": "anthropic",
        "kwargs": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        "requires_configured": "anthropic",
    },
    {
        "label": "openai",
        "kwargs": {"provider": "openai", "model": "gpt-4o-mini"},
        "requires_configured": "openai",
    },
    {
        "label": "gemini",
        "kwargs": {"provider": "gemini", "model": "gemini-flash-latest"},
        "requires_configured": "gemini",
    },
    {
        "label": "grok",
        "kwargs": {"provider": "grok", "model": "grok-4.3"},
        "requires_configured": "grok",
    },
    {
        "label": "azureopenai",
        "kwargs": {
            "provider": "azureopenai",
            "model": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            "extra_kwargs": {"azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "")},
        },
        "requires_configured": "azureopenai",
    },
    {
        "label": "openai_compatible (Ollama)",
        "kwargs": {
            "provider": "openai_compatible",
            "model": os.getenv("OPENAI_COMPATIBLE_MODEL", "gpt-oss:20b"),
            "extra_kwargs": {
                "base_url": os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:11434/v1"),
                "api_key": "ollama",
            },
        },
        # No API key gates this provider — it's attempted unconditionally
        # and any connection failure surfaces as a FAIL, not a skip.
        "requires_configured": None,
    },
]


async def run_one(session, run):
    out_dir = os.path.join(OUTPUT_ROOT, run["label"].split()[0])
    args = {
        "schema": SCHEMA,
        "sample_sizes": {"customers": 6, "orders": 10},
        "output_dir": out_dir,
        **run["kwargs"],
    }
    result = await session.call_tool("generate_from_schema", args)
    data = json.loads(result.content[0].text)

    if not data.get("ok"):
        return False, f"tool error: {data.get('error')}: {data.get('message')}"

    cust_ids = set()
    with open(os.path.join(out_dir, "customers.csv")) as f:
        for row in csv.DictReader(f):
            cust_ids.add(row["customer_id"])

    order_fks = []
    with open(os.path.join(out_dir, "orders.csv")) as f:
        for row in csv.DictReader(f):
            order_fks.append(row["customer_id"])

    orphaned = [v for v in order_fks if v not in cust_ids]
    if orphaned:
        return False, (f"FK integrity broken: {len(orphaned)}/{len(order_fks)} "
                        f"orphaned FKs {orphaned[:5]}")

    report = data.get("report", {})
    detail = (f"{len(order_fks)}/{len(order_fks)} FKs valid — "
              f"${report.get('estimated_cost_usd', 0):.4f}, "
              f"{report.get('total_time_s', 0)}s, "
              f"{report.get('model', run['kwargs'].get('model'))}")
    return True, detail


async def main():
    server_params = StdioServerParameters(command="syda-mcp")
    results = []

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            configured = set()
            try:
                providers_result = await session.call_tool("get_providers", {})
                providers_data = json.loads(providers_result.content[0].text)
                configured = set(providers_data.get("configured", []))
            except Exception:
                pass

            print("=" * 70)
            print("  provider matrix — forced provider=, real FK verification")
            print("=" * 70)

            for run in PROVIDERS:
                gate = run["requires_configured"]
                if gate and gate not in configured:
                    print(f"\n  SKIP  {run['label']:<28} (no API key configured)")
                    results.append((run["label"], None, "skipped — not configured"))
                    continue

                print(f"\n  Running {run['label']}...")
                try:
                    ok, detail = await run_one(session, run)
                except Exception as e:
                    ok, detail = False, f"{type(e).__name__}: {e}"
                results.append((run["label"], ok, detail))
                status = "PASS" if ok else "FAIL"
                print(f"  {status}  {run['label']:<28} {detail}")

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print("=" * 70)
    for label, ok, detail in results:
        status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        print(f"  {status}  {label:<28} {detail}")

    failed = [r for r in results if r[1] is False]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())

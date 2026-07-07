#!/usr/bin/env python
"""
MCP example: comprehensive test of all 5 Syda MCP tools

Runs each tool in sequence and reports pass/fail.
Tools that don't need an LLM key (validate_schema, get_providers,
infer_schema_from_db) always run. generate_from_schema is skipped
if no API key is configured.

Usage:
    pip install "syda[mcp]"
    python examples/mcp/test_all_tools.py
"""

import asyncio
import json
import os
import sqlite3
import tempfile
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

results = []


def record(tool, label, ok, detail=""):
    results.append({"tool": tool, "label": label, "ok": ok, "detail": detail})
    status = "✓" if ok else "✘"
    print(f"  {status} [{tool}] {label}")
    if detail and not ok:
        print(f"      {detail}")


async def test_get_providers(session):
    print("\n── get_providers ─────────────────────────────────────────────")
    r = await session.call_tool("get_providers", {})
    d = json.loads(r.content[0].text)
    record("get_providers", "returns ok=True", d["ok"])
    record("get_providers", "has 'providers' list", isinstance(d.get("providers"), list))
    record("get_providers", "has 'configured' list", isinstance(d.get("configured"), list))
    record("get_providers", "anthropic in providers",
           any(p["provider"] == "anthropic" for p in d["providers"]))
    record("get_providers", "azureopenai in providers",
           any(p["provider"] == "azureopenai" for p in d["providers"]))
    return d["configured"]


async def test_validate_schema(session):
    print("\n── validate_schema ───────────────────────────────────────────")

    # Valid schema
    valid = {
        "users": {
            "user_id": {"type": "integer", "primary_key": True},
            "email":   {"type": "email",   "unique": True},
        },
        "orders": {
            "order_id":  {"type": "integer", "primary_key": True},
            "user_id":   {"type": "integer",
                          "foreign_key": {"table": "users", "column": "user_id"}},
            "amount":    {"type": "float"},
        },
    }
    r = await session.call_tool("validate_schema", {"schema": valid})
    d = json.loads(r.content[0].text)
    record("validate_schema", "valid schema → ok=True", d["ok"] is True)
    record("validate_schema", "FK relationship detected",
           len(d.get("relationships", [])) == 1)
    record("validate_schema", "no errors on valid schema",
           len(d.get("errors", [])) == 0)

    # Invalid schema — missing FK table
    invalid = {
        "orders": {
            "order_id":  {"type": "integer", "primary_key": True},
            "user_id":   {"type": "integer",
                          "foreign_key": {"table": "nonexistent", "column": "id"}},
        }
    }
    r2 = await session.call_tool("validate_schema", {"schema": invalid})
    d2 = json.loads(r2.content[0].text)
    record("validate_schema", "invalid FK → ok=False", d2["ok"] is False)
    record("validate_schema", "error message present", len(d2.get("errors", [])) > 0)

    # Warning: no primary key
    warn = {"logs": {"message": {"type": "string"}}}
    r3 = await session.call_tool("validate_schema", {"schema": warn})
    d3 = json.loads(r3.content[0].text)
    record("validate_schema", "missing PK → ok=True with warning", d3["ok"] is True)
    record("validate_schema", "warning listed",
           any("primary_key" in w for w in d3.get("warnings", [])))


async def test_infer_schema_from_db(session):
    print("\n── infer_schema_from_db ──────────────────────────────────────")

    # Create a temp SQLite DB
    tmp = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(tmp)
    conn.executescript("""
        CREATE TABLE depts (
            dept_id INTEGER PRIMARY KEY,
            name    TEXT NOT NULL
        );
        CREATE TABLE emps (
            emp_id  INTEGER PRIMARY KEY,
            dept_id INTEGER NOT NULL REFERENCES depts(dept_id),
            name    TEXT,
            salary  REAL
        );
    """)
    conn.commit()
    conn.close()

    db_url = f"sqlite:///{tmp}"

    r = await session.call_tool("infer_schema_from_db", {"db_url": db_url})
    d = json.loads(r.content[0].text)
    record("infer_schema_from_db", "returns ok=True", d["ok"] is True)
    record("infer_schema_from_db", "infers both tables",
           set(d.get("tables", [])) == {"depts", "emps"})
    record("infer_schema_from_db", "FK relationship detected",
           len(d.get("relationships", [])) == 1)
    record("infer_schema_from_db", "schema ready for generation",
           "depts" in d.get("schema", {}))

    # Bad URL
    r2 = await session.call_tool("infer_schema_from_db",
                                  {"db_url": "postgresql://bad:bad@nowhere/nodb"})
    d2 = json.loads(r2.content[0].text)
    record("infer_schema_from_db", "bad URL → ok=False", d2["ok"] is False)
    record("infer_schema_from_db", "suggestion provided", "suggestion" in d2)

    os.unlink(tmp)
    return d.get("schema", {})


async def test_get_run_report(session):
    print("\n── get_run_report ────────────────────────────────────────────")
    r = await session.call_tool("get_run_report", {})
    d = json.loads(r.content[0].text)
    # get_run_report intentionally returns ok=False with a hint
    record("get_run_report", "returns hint about report field", "hint" in d)
    record("get_run_report", "hint mentions generate_from_schema",
           "generate_from_schema" in d.get("hint", ""))


async def test_generate_from_schema(session, configured_providers):
    print("\n── generate_from_schema ──────────────────────────────────────")

    if not configured_providers:
        print("  (skipped — no LLM provider configured)")
        results.append({"tool": "generate_from_schema", "label": "skipped", "ok": None})
        return

    schema = {
        "teams": {
            "team_id": {"type": "integer", "primary_key": True},
            "name":    {"type": "string"},
            "sport":   {"type": "string",
                        "enum": ["football", "basketball", "baseball", "soccer"]},
        },
        "players": {
            "player_id": {"type": "integer", "primary_key": True},
            "team_id":   {
                "type": "integer",
                "foreign_key": {"table": "teams", "column": "team_id"},
            },
            "name":      {"type": "string"},
            "position":  {"type": "string"},
            "age":       {"type": "integer", "min": 18, "max": 40},
        },
    }

    r = await session.call_tool(
        "generate_from_schema",
        {
            "schema":               schema,
            "sample_sizes":         {"teams": 4, "players": 20},
            "prompts":              {"players": "Professional sports players"},
            "preview_rows":         2,
        },
    )
    d = json.loads(r.content[0].text)
    record("generate_from_schema", "returns ok=True", d["ok"] is True,
           d.get("message", ""))
    record("generate_from_schema", "both tables in response",
           set(d.get("row_counts", {}).keys()) == {"teams", "players"})
    record("generate_from_schema", "correct row counts",
           d.get("row_counts", {}).get("teams") == 4 and
           d.get("row_counts", {}).get("players") == 20)
    record("generate_from_schema", "report included",
           "estimated_cost_usd" in d.get("report", {}))
    record("generate_from_schema", "preview rows returned",
           len(d.get("tables", {}).get("teams", [])) > 0)

    # Invalid schema → graceful error
    r2 = await session.call_tool(
        "generate_from_schema",
        {"schema": {"t": {"col": {"type": "integer",
                                   "foreign_key": {"table": "missing", "column": "id"}}}},
         "provider": configured_providers[0]},
    )
    d2 = json.loads(r2.content[0].text)
    record("generate_from_schema", "invalid schema → ok=False", d2["ok"] is False)
    record("generate_from_schema", "suggestion present", "suggestion" in d2)


async def main():
    server_params = StdioServerParameters(command="syda-mcp")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("=" * 60)
            print("Syda MCP — all tools test")
            print("=" * 60)

            configured = await test_get_providers(session)
            await test_validate_schema(session)
            inferred_schema = await test_infer_schema_from_db(session)
            await test_get_run_report(session)
            await test_generate_from_schema(session, configured)

    # Summary
    print(f"\n{'='*60}")
    passed  = sum(1 for r in results if r["ok"] is True)
    failed  = sum(1 for r in results if r["ok"] is False)
    skipped = sum(1 for r in results if r["ok"] is None)
    print(f"Results: {passed} passed  {failed} failed  {skipped} skipped")
    if failed:
        print("\nFailed:")
        for r in results:
            if r["ok"] is False:
                print(f"  ✘ [{r['tool']}] {r['label']}  {r['detail']}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

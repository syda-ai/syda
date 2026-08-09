#!/usr/bin/env python
"""
MCP example: infer_schema_from_db tool

Connects to the local SQLite healthcare demo DB (created by database_integration
examples), infers the schema, then immediately generates synthetic data from it.

No LLM key needed for inference only.
LLM key needed for the follow-up generation step.

Usage:
    pip install "syda[mcp]"
    # First create the demo DB (or use any existing SQLite DB):
    python examples/database_integration/example_load_schemas.py
    # Then run this:
    python examples/mcp/test_infer_schema.py
"""

import asyncio
import json
import os
import sqlite3
import tempfile
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DB = os.path.join(EXAMPLE_DIR, "..", "database_integration", "healthcare_demo.db")


def create_temp_db() -> str:
    """Create a minimal SQLite DB for testing if demo DB doesn't exist."""
    tmp = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(tmp)
    conn.executescript("""
        CREATE TABLE departments (
            dept_id   INTEGER PRIMARY KEY,
            name      TEXT NOT NULL,
            location  TEXT
        );
        CREATE TABLE employees (
            emp_id    INTEGER PRIMARY KEY,
            dept_id   INTEGER NOT NULL REFERENCES departments(dept_id),
            first_name TEXT NOT NULL,
            last_name  TEXT NOT NULL,
            email      TEXT UNIQUE,
            salary     REAL,
            hire_date  TEXT
        );
        CREATE TABLE projects (
            project_id INTEGER PRIMARY KEY,
            dept_id    INTEGER NOT NULL REFERENCES departments(dept_id),
            name       TEXT NOT NULL,
            status     TEXT CHECK(status IN ('active','completed','paused'))
        );
        INSERT INTO departments VALUES (1,'Engineering','Floor 3'),(2,'Marketing','Floor 1');
        INSERT INTO employees VALUES
            (1,1,'Alice','Smith','a@co.com',90000,'2022-01-10'),
            (2,2,'Bob','Jones','b@co.com',75000,'2021-06-15');
        INSERT INTO projects VALUES (1,1,'API Rewrite','active'),(2,2,'Campaign Q3','completed');
    """)
    conn.commit()
    conn.close()
    return tmp


async def main():
    # Determine which DB to use
    if os.path.exists(DEMO_DB):
        db_path = DEMO_DB
        print(f"Using healthcare demo DB: {db_path}")
    else:
        db_path = create_temp_db()
        print(f"Demo DB not found — created temp DB: {db_path}")

    db_url = f"sqlite:///{db_path}"

    server_params = StdioServerParameters(command="syda-mcp")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ── Step 1: infer schema ──────────────────────────────────────────
            print("\n" + "=" * 60)
            print("Step 1: infer_schema_from_db")
            print("=" * 60)

            result = await session.call_tool(
                "infer_schema_from_db",
                {"db_url": db_url},
            )
            infer_data = json.loads(result.content[0].text)

            if not infer_data["ok"]:
                print(f"ERROR: {infer_data['message']}")
                return

            print(f"\n✓ {infer_data['message']}")
            print(f"\nTables: {', '.join(infer_data['tables'])}")

            if infer_data["relationships"]:
                print("\nRelationships:")
                for r in infer_data["relationships"]:
                    print(f"  {r['child']} → {r['parent']}")

            print("\nInferred schema (ready for generate_from_schema):")
            for table, cols in infer_data["schema"].items():
                col_names = [k for k in cols if not k.startswith("_")]
                print(f"  {table}: {', '.join(col_names)}")

            # ── Step 2: validate the inferred schema ──────────────────────────
            print("\n" + "=" * 60)
            print("Step 2: validate_schema (on inferred schema)")
            print("=" * 60)

            val = await session.call_tool(
                "validate_schema",
                {"schema": infer_data["schema"]},
            )
            val_data = json.loads(val.content[0].text)
            print(f"\n✓ {val_data['message']}")
            if val_data["warnings"]:
                for w in val_data["warnings"]:
                    print(f"  [warning] {w}")

            # ── Step 3: generate (only if API key available) ──────────────────
            print("\n" + "=" * 60)
            print("Step 3: generate_from_schema (using inferred schema)")
            print("=" * 60)

            providers_result = await session.call_tool("get_providers", {})
            providers_data = json.loads(providers_result.content[0].text)

            if not providers_data["configured"]:
                print("\nNo LLM provider configured — skipping generation step.")
                print("Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GROK_API_KEY to enable.")
                return

            print(f"\nGenerating with auto-detected provider: {providers_data['configured'][0]}")

            sample_sizes = {t: 20 for t in infer_data["tables"]}

            gen = await session.call_tool(
                "generate_from_schema",
                {
                    "schema":        infer_data["schema"],
                    "sample_sizes":  sample_sizes,
                    "preview_rows":  3,
                },
            )
            gen_data = json.loads(gen.content[0].text)

            if not gen_data["ok"]:
                print(f"ERROR: {gen_data['message']}")
                return

            print(f"\n✓ {gen_data['message']}")
            for table, count in gen_data["row_counts"].items():
                print(f"  {table}: {count} rows")

            report = gen_data.get("report", {})
            if report:
                print(f"\n  Cost : ${report.get('estimated_cost_usd', 0):.4f}")
                print(f"  Time : {report.get('total_time_s', '?')}s")

            print("\n── Sample rows ─────────────────────────────────────────────")
            for table, rows in gen_data["tables"].items():
                print(f"\n{table}:")
                for row in rows:
                    print(f"  {row}")

    # Cleanup temp DB if we created it
    if db_path != DEMO_DB and os.path.exists(db_path):
        os.unlink(db_path)


if __name__ == "__main__":
    asyncio.run(main())

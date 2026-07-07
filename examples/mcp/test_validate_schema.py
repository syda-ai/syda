#!/usr/bin/env python
"""
MCP example: validate_schema tool

Tests schema validation — catches errors before making any LLM calls.
No API key required for this tool.

Usage:
    pip install "syda[mcp]"
    python examples/mcp/test_validate_schema.py
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

CASES = [
    # ── Valid schemas ──────────────────────────────────────────────────────────
    {
        "label": "Simple single-table schema",
        "expect_ok": True,
        "schema": {
            "users": {
                "user_id": {"type": "integer", "primary_key": True},
                "email":   {"type": "email",   "unique": True},
                "name":    {"type": "string"},
                "plan":    {"type": "string",  "enum": ["free", "pro", "enterprise"]},
                "created": {"type": "date"},
            }
        },
    },
    {
        "label": "FK chain: customers → orders → order_items",
        "expect_ok": True,
        "schema": {
            "customers": {
                "customer_id": {"type": "integer", "primary_key": True},
                "name":        {"type": "string"},
            },
            "orders": {
                "order_id":   {"type": "integer", "primary_key": True},
                "customer_id": {
                    "type": "integer",
                    "foreign_key": {"table": "customers", "column": "customer_id"},
                },
                "amount":    {"type": "float", "min": 1.0, "max": 5000.0},
            },
            "order_items": {
                "item_id":  {"type": "integer", "primary_key": True},
                "order_id": {
                    "type": "integer",
                    "foreign_key": {"table": "orders", "column": "order_id"},
                },
                "quantity": {"type": "integer", "min": 1, "max": 10},
            },
        },
    },
    {
        "label": "Healthcare schema with multiple FK relationships",
        "expect_ok": True,
        "schema": {
            "patients": {
                "patient_id":  {"type": "integer", "primary_key": True},
                "dob":         {"type": "date"},
                "gender":      {"type": "string", "enum": ["M", "F", "Other"]},
            },
            "providers": {
                "provider_id": {"type": "integer", "primary_key": True},
                "specialty":   {"type": "string"},
            },
            "encounters": {
                "encounter_id": {"type": "integer", "primary_key": True},
                "patient_id":   {
                    "type": "integer",
                    "foreign_key": {"table": "patients", "column": "patient_id"},
                },
                "provider_id":  {
                    "type": "integer",
                    "foreign_key": {"table": "providers", "column": "provider_id"},
                },
                "encounter_date": {"type": "date"},
                "diagnosis_code": {"type": "string"},
            },
        },
    },

    # ── Invalid schemas ────────────────────────────────────────────────────────
    {
        "label": "FK references non-existent table",
        "expect_ok": False,
        "schema": {
            "orders": {
                "order_id":   {"type": "integer", "primary_key": True},
                "customer_id": {
                    "type": "integer",
                    "foreign_key": {"table": "customers", "column": "customer_id"},
                },
            }
        },
    },
    {
        "label": "FK definition missing 'column' key",
        "expect_ok": False,
        "schema": {
            "customers": {
                "customer_id": {"type": "integer", "primary_key": True},
            },
            "orders": {
                "order_id":   {"type": "integer", "primary_key": True},
                "customer_id": {
                    "type": "integer",
                    "foreign_key": {"table": "customers"},  # missing "column"
                },
            },
        },
    },

    # ── Warning cases (ok=True but with warnings) ──────────────────────────────
    {
        "label": "No primary key declared (warning only)",
        "expect_ok": True,
        "schema": {
            "logs": {
                "message":   {"type": "string"},
                "logged_at": {"type": "date"},
            }
        },
    },
    {
        "label": "Unknown column type (warning only)",
        "expect_ok": True,
        "schema": {
            "events": {
                "event_id": {"type": "integer", "primary_key": True},
                "payload":  {"type": "jsonb"},   # unsupported, warns not errors
            }
        },
    },
]


async def main():
    server_params = StdioServerParameters(command="syda-mcp")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("=" * 60)
            print("validate_schema — test suite")
            print("=" * 60)

            passed = failed = 0

            for case in CASES:
                result = await session.call_tool(
                    "validate_schema", {"schema": case["schema"]}
                )
                data = json.loads(result.content[0].text)

                ok      = data["ok"]
                errors  = data.get("errors", [])
                warns   = data.get("warnings", [])
                rels    = data.get("relationships", [])
                correct = (ok == case["expect_ok"])

                status = "✓ PASS" if correct else "✘ FAIL"
                if correct:
                    passed += 1
                else:
                    failed += 1

                print(f"\n{status}  {case['label']}")
                print(f"  ok={ok}  errors={len(errors)}  warnings={len(warns)}  "
                      f"relationships={len(rels)}")
                if errors:
                    for e in errors:
                        print(f"  [error]   {e}")
                if warns:
                    for w in warns:
                        print(f"  [warning] {w}")
                if rels:
                    for r in rels:
                        print(f"  [fk]      {r['child']} → {r['parent']}")
                if not correct:
                    print(f"  !! Expected ok={case['expect_ok']}, got ok={ok}")

            print(f"\n{'='*60}")
            print(f"Results: {passed} passed, {failed} failed")
            print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

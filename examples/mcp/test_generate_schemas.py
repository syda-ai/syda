#!/usr/bin/env python
"""
MCP example: generate_from_schema tool — multiple schemas

Tests generation with different domain schemas:
  1. E-commerce  (customers → orders → order_items, products)
  2. SaaS        (companies → users → subscriptions)
  3. Healthcare  (patients → encounters → diagnoses)

Runs whichever LLM provider is configured (auto-detected).

Usage:
    pip install "syda[mcp]"
    export ANTHROPIC_API_KEY=sk-ant-...   # or any provider key
    python examples/mcp/test_generate_schemas.py
"""

import asyncio
import json
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SCHEMAS = {
    "E-commerce": {
        "schema": {
            "customers": {
                "customer_id":  {"type": "integer", "primary_key": True},
                "email":        {"type": "email",   "unique": True},
                "name":         {"type": "string"},
                "country":      {"type": "string",
                                 "enum": ["US","UK","DE","FR","IN","CA","AU"]},
                "loyalty_tier": {"type": "string",
                                 "enum": ["Bronze","Silver","Gold","Platinum"],
                                 "probabilities": [0.40, 0.30, 0.20, 0.10]},
                "signup_date":  {"type": "date"},
            },
            "products": {
                "product_id":   {"type": "integer", "primary_key": True},
                "name":         {"type": "string"},
                "category":     {"type": "string",
                                 "enum": ["Electronics","Clothing","Home","Sports"]},
                "price":        {"type": "float", "min": 5.0, "max": 500.0},
            },
            "orders": {
                "order_id":     {"type": "integer", "primary_key": True},
                "customer_id":  {
                    "type": "integer",
                    "foreign_key": {"table": "customers", "column": "customer_id"},
                },
                "order_date":   {"type": "date"},
                "status":       {"type": "string",
                                 "enum": ["pending","shipped","delivered","cancelled"],
                                 "probabilities": [0.10, 0.20, 0.60, 0.10]},
                "total_amount": {"type": "float", "min": 5.0, "max": 2000.0},
            },
            "order_items": {
                "item_id":      {"type": "integer", "primary_key": True},
                "order_id":     {
                    "type": "integer",
                    "foreign_key": {"table": "orders", "column": "order_id"},
                },
                "product_id":   {
                    "type": "integer",
                    "foreign_key": {"table": "products", "column": "product_id"},
                },
                "quantity":     {"type": "integer", "min": 1, "max": 5},
                "unit_price":   {"type": "float",   "min": 5.0, "max": 500.0},
            },
        },
        "sample_sizes": {"customers": 20, "products": 10, "orders": 40, "order_items": 80},
        "prompts": {
            "customers": "Diverse global e-commerce customers",
            "orders":    "Orders from the past 2 years, mostly delivered",
        },
    },

    "SaaS": {
        "schema": {
            "companies": {
                "company_id":  {"type": "integer", "primary_key": True},
                "name":        {"type": "string"},
                "industry":    {"type": "string",
                                "enum": ["Tech","Finance","Healthcare","Retail","Education"]},
                "size":        {"type": "string",
                                "enum": ["startup","smb","enterprise"],
                                "probabilities": [0.40, 0.40, 0.20]},
            },
            "users": {
                "user_id":    {"type": "integer", "primary_key": True},
                "company_id": {
                    "type": "integer",
                    "foreign_key": {"table": "companies", "column": "company_id"},
                },
                "email":      {"type": "email", "unique": True},
                "role":       {"type": "string",
                               "enum": ["admin","member","viewer"],
                               "probabilities": [0.20, 0.60, 0.20]},
                "created_at": {"type": "date"},
            },
            "subscriptions": {
                "sub_id":       {"type": "integer", "primary_key": True},
                "company_id":   {
                    "type": "integer",
                    "foreign_key": {"table": "companies", "column": "company_id"},
                },
                "plan":         {"type": "string",
                                 "enum": ["free","starter","pro","enterprise"],
                                 "probabilities": [0.30, 0.30, 0.25, 0.15]},
                "mrr_usd":      {"type": "float", "min": 0.0, "max": 5000.0},
                "status":       {"type": "string",
                                 "enum": ["active","churned","trial"],
                                 "probabilities": [0.70, 0.20, 0.10]},
                "start_date":   {"type": "date"},
            },
        },
        "sample_sizes": {"companies": 10, "users": 30, "subscriptions": 10},
        "prompts": {
            "companies": "B2B SaaS customers across various industries",
        },
    },

    "Healthcare": {
        "schema": {
            "patients": {
                "patient_id":  {"type": "integer", "primary_key": True},
                "dob":         {"type": "date"},
                "gender":      {"type": "string",
                                "enum": ["M","F","Other"],
                                "probabilities": [0.49, 0.49, 0.02]},
                "blood_type":  {"type": "string",
                                "enum": ["A+","A-","B+","B-","O+","O-","AB+","AB-"]},
            },
            "providers": {
                "provider_id": {"type": "integer", "primary_key": True},
                "specialty":   {"type": "string",
                                "enum": ["Cardiology","Oncology","Neurology",
                                         "Orthopedics","General Practice"]},
                "npi":         {"type": "string"},
            },
            "encounters": {
                "encounter_id":   {"type": "integer", "primary_key": True},
                "patient_id":     {
                    "type": "integer",
                    "foreign_key": {"table": "patients", "column": "patient_id"},
                },
                "provider_id":    {
                    "type": "integer",
                    "foreign_key": {"table": "providers", "column": "provider_id"},
                },
                "encounter_date": {"type": "date"},
                "encounter_type": {"type": "string",
                                   "enum": ["inpatient","outpatient","emergency","telehealth"]},
                "diagnosis_code": {"type": "string"},
                "notes":          {"type": "text"},
            },
        },
        "sample_sizes": {"patients": 15, "providers": 5, "encounters": 30},
        "prompts": {
            "patients":   "US patients, age range 18-85",
            "encounters": "Realistic clinical encounters with ICD-10 diagnosis codes",
        },
    },
}


async def run_schema(session, name, config):
    print(f"\n{'='*60}")
    print(f"  Schema: {name}")
    print(f"{'='*60}")

    total_rows = sum(config["sample_sizes"].values())
    print(f"  Tables : {', '.join(config['schema'].keys())}")
    print(f"  Rows   : {total_rows} total")

    result = await session.call_tool(
        "generate_from_schema",
        {
            "schema":        config["schema"],
            "sample_sizes":  config["sample_sizes"],
            "prompts":       config.get("prompts", {}),
            "preview_rows":  2,
        },
    )
    data = json.loads(result.content[0].text)

    if not data["ok"]:
        print(f"  ✘ FAILED: {data['message']}")
        return False

    print(f"  ✓ {data['message']}")

    report = data.get("report", {})
    if report:
        print(f"  Cost   : ${report.get('estimated_cost_usd', 0):.4f}")
        print(f"  Calls  : {report.get('total_llm_calls', '?')}")
        print(f"  Time   : {report.get('total_time_s', '?')}s")

    print("\n  Sample rows:")
    for table, rows in data["tables"].items():
        if rows:
            print(f"    [{table}] {rows[0]}")

    return True


async def main():
    server_params = StdioServerParameters(command="syda-mcp")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Check provider
            prov = await session.call_tool("get_providers", {})
            prov_data = json.loads(prov.content[0].text)
            if not prov_data["configured"]:
                print("No LLM provider configured. Set an API key and retry.")
                return
            print(f"Provider: {prov_data['configured'][0]}")

            passed = failed = 0
            for name, config in SCHEMAS.items():
                ok = await run_schema(session, name, config)
                if ok:
                    passed += 1
                else:
                    failed += 1

            print(f"\n{'='*60}")
            print(f"Results: {passed} passed, {failed} failed across {len(SCHEMAS)} schemas")
            print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

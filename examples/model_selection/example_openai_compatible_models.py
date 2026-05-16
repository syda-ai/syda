import os
import sys
from dotenv import load_dotenv

load_dotenv()

base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:11434/v1")
api_key  = os.environ.get("OPENAI_COMPATIBLE_API_KEY", "ollama")
model    = os.environ.get("OPENAI_COMPATIBLE_MODEL", "llama3.2")

try:
    import httpx
    resp = httpx.get(base_url.rstrip("/v1").rstrip("/"), timeout=3)
except Exception as e:
    print(f"OpenAI-compatible server not reachable at {base_url}: {e}")
    print("Skipping openai_compatible example (set OPENAI_COMPATIBLE_BASE_URL to enable).")
    sys.exit(0)

from syda.generate import SyntheticDataGenerator
from syda.schemas import ModelConfig

schemas = {
    "Company": {
        "company_id": {"type": "number", "description": "Unique company identifier"},
        "name":       {"type": "text",   "description": "Company name"},
        "industry":   {"type": "text",   "description": "Industry sector"},
        "founded":    {"type": "date",   "description": "Date the company was founded"},
        "revenue":    {"type": "number", "description": "Annual revenue in USD"},
    },
    "Product": {
        "product_id": {"type": "number",      "description": "Unique product identifier"},
        "company_id": {"type": "foreign_key", "description": "Company that makes this product",
                       "references": {"schema": "Company", "field": "company_id"}},
        "name":       {"type": "text",        "description": "Product name"},
        "category":   {"type": "text",        "description": "Product category"},
        "price":      {"type": "number",      "description": "Price in USD"},
    },
}

prompts = {
    "Company": "Generate realistic synthetic technology company records.",
    "Product": "Generate realistic synthetic software product records.",
}

print(f"----- Testing openai_compatible: {base_url} | model: {model} -----")

model_config = ModelConfig(
    provider="openai_compatible",
    model_name=model,
    extra_kwargs={
        "base_url":      base_url,
        "api_key":       api_key,
        "response_mode": "markdown",
    },
)

generator = SyntheticDataGenerator(model_config=model_config)

output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output",
    "test_openai_compatible",
    model.replace(":", "-"),
)

results = generator.generate_for_schemas(
    schemas=schemas,
    prompts=prompts,
    sample_sizes={"Company": 5, "Product": 10},
    output_dir=output_dir,
)
print(f"Data saved to {output_dir}")

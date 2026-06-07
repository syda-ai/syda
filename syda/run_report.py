"""Runtime telemetry for a generate_for_schemas() call.

RunReport is returned alongside generated data when return_report=True.
It gives per-column strategy breakdown, LLM call counts, token usage,
estimated cost, and cache hit/miss status.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

# Per-MTok pricing (input_price, output_price).
_PRICING: Dict[str, tuple] = {
    "claude-haiku-4-5":    (0.80,   4.00),
    "claude-sonnet-4-6":   (3.00,  15.00),
    "claude-opus-4-7":    (15.00,  75.00),
    "gpt-4o-mini":         (0.15,   0.60),
    "gpt-4o":              (2.50,  10.00),
    "gemini-1.5-flash":    (0.075,  0.30),
    "gemini-1.5-pro":      (1.25,   5.00),
}


def _estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    for key, (ip, op) in _PRICING.items():
        if key in model:
            return in_tok / 1_000_000 * ip + out_tok / 1_000_000 * op
    return 0.0


@dataclass
class ColumnReport:
    """Per-column generation telemetry."""
    strategy: str
    """One of: 'fk_sampler' | 'codegen_simple' | 'codegen_semantic' | 'direct_llm'"""
    from_cache: bool = False
    generated_function: Optional[str] = None  # Python source; codegen_simple only
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class TableReport:
    """Per-table generation telemetry."""
    row_count: int = 0
    duration_s: float = 0.0
    mode: str = ""          # "direct" | "codegen"
    schema_hash: str = ""
    columns: Dict[str, ColumnReport] = field(default_factory=dict)

    @property
    def from_cache(self) -> bool:
        if self.mode != "codegen":
            return False
        simple = [c for c in self.columns.values() if c.strategy == "codegen_simple"]
        return bool(simple) and all(c.from_cache for c in simple)

    @property
    def llm_calls(self) -> int:
        return sum(c.llm_calls for c in self.columns.values())

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.columns.values())

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.columns.values())


@dataclass
class RunReport:
    """Full-run telemetry returned by generate_for_schemas(return_report=True)."""
    model: str = ""
    tables: Dict[str, TableReport] = field(default_factory=dict)
    total_duration_s: float = 0.0

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.tables.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.tables.values())

    @property
    def estimated_cost_usd(self) -> float:
        return _estimate_cost(self.model, self.total_input_tokens, self.total_output_tokens)

    def save_html_report(self, path: str) -> None:
        """Write a self-contained HTML report to *path*."""
        import os, html as _html

        _STRATEGY_BADGE = {
            "fk_sampler":      ("#e8f4fd", "#2980b9", "FK sampler"),
            "codegen_simple":  ("#eafaf1", "#27ae60", "Code-gen"),
            "codegen_semantic":("#fef9e7", "#f39c12", "Semantic LLM"),
            "direct_llm":      ("#f5eef8", "#8e44ad", "Direct LLM"),
        }

        def badge(strategy: str) -> str:
            bg, fg, label = _STRATEGY_BADGE.get(strategy, ("#eee", "#333", strategy))
            return (f'<span style="background:{bg};color:{fg};padding:2px 8px;'
                    f'border-radius:10px;font-size:0.8em;font-weight:600">{label}</span>')

        def cache_pill(yes: bool) -> str:
            if yes:
                return '<span style="background:#d5f5e3;color:#1e8449;padding:2px 8px;border-radius:10px;font-size:0.8em;font-weight:600">HIT</span>'
            return '<span style="background:#fdebd0;color:#a04000;padding:2px 8px;border-radius:10px;font-size:0.8em;font-weight:600">MISS</span>'

        rows_total = sum(t.row_count for t in self.tables.values())
        calls_total = sum(t.llm_calls for t in self.tables.values())
        cost = self.estimated_cost_usd
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # ── Summary table rows ───────────────────────────────────────────
        summary_rows = ""
        for name, t in self.tables.items():
            cache_cell = cache_pill(t.from_cache) if t.mode == "codegen" else ""
            summary_rows += f"""
            <tr>
              <td><a href="#{name}" style="color:#2c3e50;font-weight:600">{_html.escape(name)}</a></td>
              <td style="text-align:right">{t.row_count:,}</td>
              <td><span style="font-variant:small-caps">{t.mode}</span></td>
              <td style="text-align:right">{t.llm_calls:,}</td>
              <td style="text-align:right">{t.input_tokens:,}</td>
              <td style="text-align:right">{t.output_tokens:,}</td>
              <td style="text-align:right">{t.duration_s:.1f}s</td>
              <td>{cache_cell}</td>
            </tr>"""

        # ── Per-table detail sections ────────────────────────────────────
        table_details = ""
        for name, t in self.tables.items():
            col_rows = ""
            for col_name, c in t.columns.items():
                fn_block = ""
                if c.generated_function:
                    escaped = _html.escape(c.generated_function)
                    fn_block = (
                        f'<details style="margin-top:6px">'
                        f'<summary style="cursor:pointer;color:#555;font-size:0.85em">View generated function</summary>'
                        f'<pre style="background:#1e1e1e;color:#d4d4d4;padding:12px;border-radius:6px;'
                        f'overflow-x:auto;font-size:0.82em;margin-top:6px">{escaped}</pre>'
                        f'</details>'
                    )
                cache_col = ""
                if c.strategy == "codegen_simple":
                    cache_col = cache_pill(c.from_cache)
                col_rows += f"""
                <tr>
                  <td style="font-family:monospace">{_html.escape(col_name)}</td>
                  <td>{badge(c.strategy)}</td>
                  <td style="text-align:right">{c.llm_calls}</td>
                  <td style="text-align:right">{c.input_tokens:,}</td>
                  <td style="text-align:right">{c.output_tokens:,}</td>
                  <td>{cache_col}</td>
                  <td>{fn_block}</td>
                </tr>"""

            schema_hash_pill = (
                f'<code style="background:#f0f0f0;padding:2px 6px;border-radius:4px;font-size:0.8em">'
                f'{t.schema_hash}</code>' if t.schema_hash else ""
            )
            cache_header = cache_pill(t.from_cache) if t.mode == "codegen" else ""

            table_details += f"""
            <section id="{name}" style="margin-bottom:40px">
              <h2 style="border-bottom:2px solid #e0e0e0;padding-bottom:8px;color:#2c3e50">
                {_html.escape(name)}
                <span style="font-size:0.6em;font-weight:400;margin-left:12px">{t.row_count:,} rows &nbsp;·&nbsp;
                  <span style="font-variant:small-caps">{t.mode}</span> &nbsp;·&nbsp;
                  {t.duration_s:.1f}s</span>
                <span style="margin-left:12px">{cache_header}</span>
                <span style="margin-left:8px;font-size:0.55em;font-weight:400">{schema_hash_pill}</span>
              </h2>
              <table style="width:100%;border-collapse:collapse;font-size:0.9em">
                <thead>
                  <tr style="background:#f7f9fc;text-transform:uppercase;font-size:0.75em;letter-spacing:0.05em">
                    <th style="text-align:left;padding:8px">Column</th>
                    <th style="text-align:left;padding:8px">Strategy</th>
                    <th style="text-align:right;padding:8px">Calls</th>
                    <th style="text-align:right;padding:8px">In tok</th>
                    <th style="text-align:right;padding:8px">Out tok</th>
                    <th style="padding:8px">Cache</th>
                    <th style="padding:8px">Function</th>
                  </tr>
                </thead>
                <tbody>{col_rows}</tbody>
              </table>
            </section>"""

        cost_str = f"${cost:.4f}" if cost else "n/a"
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>syda run report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           margin: 0; padding: 24px 40px; color: #2c3e50; background: #fff; }}
    table {{ border-collapse: collapse; }}
    th, td {{ padding: 8px 12px; border-bottom: 1px solid #eee; vertical-align: top; }}
    thead tr {{ background: #f7f9fc; }}
    tr:hover {{ background: #fafafa; }}
    a {{ text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1 style="margin-bottom:4px">syda run report</h1>
  <p style="color:#888;margin-top:0">{generated_at} &nbsp;·&nbsp; model: <code>{_html.escape(self.model)}</code></p>

  <div style="display:flex;gap:24px;margin:20px 0">
    <div style="background:#f7f9fc;border-radius:8px;padding:16px 24px;min-width:120px;text-align:center">
      <div style="font-size:2em;font-weight:700">{rows_total:,}</div>
      <div style="color:#888;font-size:0.85em">total rows</div>
    </div>
    <div style="background:#f7f9fc;border-radius:8px;padding:16px 24px;min-width:120px;text-align:center">
      <div style="font-size:2em;font-weight:700">{calls_total}</div>
      <div style="color:#888;font-size:0.85em">LLM calls</div>
    </div>
    <div style="background:#f7f9fc;border-radius:8px;padding:16px 24px;min-width:120px;text-align:center">
      <div style="font-size:2em;font-weight:700">{self.total_input_tokens:,}</div>
      <div style="color:#888;font-size:0.85em">input tokens</div>
    </div>
    <div style="background:#f7f9fc;border-radius:8px;padding:16px 24px;min-width:120px;text-align:center">
      <div style="font-size:2em;font-weight:700">{self.total_output_tokens:,}</div>
      <div style="color:#888;font-size:0.85em">output tokens</div>
    </div>
    <div style="background:#f7f9fc;border-radius:8px;padding:16px 24px;min-width:120px;text-align:center">
      <div style="font-size:2em;font-weight:700">{cost_str}</div>
      <div style="color:#888;font-size:0.85em">est. cost</div>
    </div>
    <div style="background:#f7f9fc;border-radius:8px;padding:16px 24px;min-width:120px;text-align:center">
      <div style="font-size:2em;font-weight:700">{self.total_duration_s:.1f}s</div>
      <div style="color:#888;font-size:0.85em">total time</div>
    </div>
  </div>

  <h2 style="border-bottom:2px solid #e0e0e0;padding-bottom:8px">Summary</h2>
  <table style="width:100%">
    <thead>
      <tr style="text-transform:uppercase;font-size:0.75em;letter-spacing:0.05em">
        <th style="text-align:left">Table</th>
        <th style="text-align:right">Rows</th>
        <th style="text-align:left">Mode</th>
        <th style="text-align:right">LLM calls</th>
        <th style="text-align:right">In tokens</th>
        <th style="text-align:right">Out tokens</th>
        <th style="text-align:right">Time</th>
        <th>Cache</th>
      </tr>
    </thead>
    <tbody>{summary_rows}</tbody>
    <tfoot>
      <tr style="font-weight:700;border-top:2px solid #ccc">
        <td>TOTAL</td>
        <td style="text-align:right">{rows_total:,}</td>
        <td></td>
        <td style="text-align:right">{calls_total}</td>
        <td style="text-align:right">{self.total_input_tokens:,}</td>
        <td style="text-align:right">{self.total_output_tokens:,}</td>
        <td></td>
        <td></td>
      </tr>
    </tfoot>
  </table>

  <h2 style="border-bottom:2px solid #e0e0e0;padding-bottom:8px;margin-top:40px">Per-table details</h2>
  {table_details}
</body>
</html>"""

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[syda] Run report saved → {path}")

    def print_summary(self) -> None:
        w = 70
        print(f"\n{'─' * w}")
        print(f"{'Table':<16} {'Rows':>6}  {'Mode':<8}  {'Calls':>5}  "
              f"{'In tok':>8}  {'Out tok':>8}  {'Cache'}")
        print(f"{'─' * w}")
        for name, t in self.tables.items():
            cache_tag = "HIT" if t.from_cache else ""
            print(
                f"{name:<16} {t.row_count:>6}  {t.mode:<8}  {t.llm_calls:>5}  "
                f"{t.input_tokens:>8,}  {t.output_tokens:>8,}  {cache_tag}"
            )
        print(f"{'─' * w}")
        total_rows = sum(t.row_count for t in self.tables.values())
        total_calls = sum(t.llm_calls for t in self.tables.values())
        print(
            f"{'TOTAL':<16} {total_rows:>6}  {'':8}  {total_calls:>5}  "
            f"{self.total_input_tokens:>8,}  {self.total_output_tokens:>8,}"
        )
        cost = self.estimated_cost_usd
        if cost:
            print(f"\nEstimated cost : ${cost:.4f}  (model: {self.model})")
        print(f"Total time     : {self.total_duration_s:.1f}s")
        print(f"{'─' * w}\n")

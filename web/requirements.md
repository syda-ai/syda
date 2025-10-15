## Syda Web UI Requirements (Phase 1)

### Overview
- Build a local-first Web UI to configure models, author schemas/templates, run generations, visualize dependencies, validate integrity, and browse/download results.
- Phase 1 must include the Dependency Graph.

### Tech stack (target)
- UI: React + TypeScript (Vite or Next.js SPA mode), React Query/Fetch for data, React Flow for graphs.
- API: Python (FastAPI) under `web/api` reusing core library (`syda`) for dependency analysis and generation.
- Output viewing: inline CSV/JSON preview; PDF preview for documents.

### Scope (Phase 1)
- Workspace & files
  - Open the current repo as workspace; operate on schemas/templates and outputs within it.
  - Safe writes with overwrite confirmation.
- Schema manager (structured)
  - Import/edit YAML/JSON schemas with inline validation and error messages.
  - Show FK declarations (`__foreign_keys__`, field `references`, `_id` heuristic) and constraints.
- Model configuration
  - Select provider (Anthropic/OpenAI/Gemini), model name, temperature, max tokens.
  - API key status indicator (read from environment; do not persist secrets in UI).
- Run wizard
  - Select schemas; set per-schema sample sizes and prompts; choose output format (CSV/JSON) and output directory.
  - Show computed generation order prior to run; live log during run.
- Results browser
  - Preview datasets (paged, sortable); download CSV/JSON; show simple stats.
  - Preview generated PDFs for template outputs (where available).
- Integrity & validation
  - FK integrity report; basic constraints (not_null, unique) checks with counts.
- Dependency Graph (mandatory)
  - Visualize schemas and dependencies; show generation order; highlight cycles/missing refs.
- Examples
  - Load-and-run from existing `examples/` to produce sample outputs quickly.

### Dependency Graph (Phase 1) — Detailed Requirements
- Data sources
  - Use `SchemaLoader` to load all provided schemas (YAML/JSON/dict) and detect template schemas.
  - Collect dependencies from: `__foreign_keys__`, field-level `references`, `_id` heuristic, and `__depends_on__` for templates.
- API
  - Endpoint: `POST /api/dependencies/graph`
    - Body: optional inline schemas (name → content) and/or paths; if empty, read default schema locations.
    - Response model:
    ```json
    {
      "nodes": [
        { "id": "Product", "label": "Product", "kind": "structured" | "template", "rank": 2 }
      ],
      "edges": [
        {
          "id": "products.category_id->Category.id",
          "source": "Product",
          "target": "Category",
          "kind": "foreign_key" | "depends_on",
          "fks": [{ "from": "category_id", "to": "id" }],
          "status": "ok" | "missing" | "cycle"
        }
      ],
      "order": ["Category", "Product", "ProductCatalog"],
      "issues": { "cycles": [["A","B","A"]], "missing": ["Order.customer_id -> Customer.id"] }
    }
    ```
  - Optional: `GET /api/dependencies/graph` to scan workspace defaults.
- UI/UX
  - Graph rendering with React Flow + Dagre layout; pan/zoom, fit-to-view, drag nodes.
  - Node styles: blue (structured), purple (template); show generation order badge.
  - Edge styles: solid (FK), dotted (depends_on); red dashed for missing; orange for edges in cycles.
  - Hover edge tooltip: show FK columns mapping; click node opens details (fields, constraints, inbound/outbound deps).
  - Toggles: show/hide template edges, FK edges, self-references.
- Validation overlays
  - Cycles detected and listed; nodes/edges in cycles highlighted.
  - Missing references listed; edges marked "missing".
- Performance
  - Handle 50–100 nodes smoothly; debounce recompute on schema edits; lazy compute on first open.

### Non-functional requirements
- Local-first, no authentication in Phase 1.
- Do not persist API keys server-side; rely on environment variables.
- Accessibility: keyboard navigation for graph controls; sufficient color contrast.
- Telemetry: none by default.

### API surface (Phase 1)
- `POST /api/dependencies/graph` — returns graph JSON and generation order.
- `POST /api/run` — run generation for selected schemas with prompts/sample sizes; returns job id and streams logs.
- `GET /api/results/:schema` — paged preview data.
- `GET /api/files/:path` — download outputs (CSV/JSON/PDF).

### UI structure (proposed)
- `web/ui/src/features/deps/DependencyGraph.tsx` — graph view.
- `web/ui/src/features/deps/DetailsPanel.tsx` — node/edge details.
- `web/ui/src/features/deps/useDependencyGraph.ts` — fetch/transform hook.
- `web/ui/src/features/schemas/*` — schema editor/import/validation.
- `web/ui/src/features/run/*` — run wizard, live logs.
- `web/ui/src/features/results/*` — results browser and downloads.

### Acceptance criteria (Phase 1)
- Graph shows all schemas with FK and template dependencies; generation order badges rendered when DAG exists.
- Cycles and missing references clearly indicated both on-graph and in an issues list.
- Schema edits trigger graph refresh within 1–2 seconds (debounced) without page reload.
- Users can configure provider/model parameters and run generation; progress/logs visible; outputs downloadable.
- FK integrity report available after run (counts of violations) and displayed in UI.

### Out of scope (defer to Phase 2)
- User auth/roles, API tokens, audit logs.
- Scheduling & background job queues beyond simple per-run process.
- Cost tracking and token usage dashboards.
- Full template live preview with sample data (basic preview is acceptable later).



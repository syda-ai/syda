import { useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router";
import { LiveConnections, type SchemaSource, useLiveProfiles } from "./live-connections";
import { SchemaExplorer } from "./schema-explorer";
import { Badge } from "~/components/ui/badge";
import { Button, buttonVariants } from "~/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "~/components/ui/card";
import { Input } from "~/components/ui/input";
import { Spinner } from "~/components/ui/spinner";
import type { GenerationJob, ScenarioConfiguration } from "~/lib/studio-types";

type Connections = ReturnType<typeof useLiveProfiles>;

export function ConnectionsWorkspace({ connections, source, onSource }: { connections: Connections; source: SchemaSource | null; onSource: (source: SchemaSource | null) => void }) {
  return <WorkspaceSection title="Connections" description="Connect to a database and bring its table structure into your next chat prompt.">
    <Card><CardHeader><CardTitle>Database connections</CardTitle><CardDescription>Inspect PostgreSQL, MySQL, or Oracle structure. Profiles last while this chat page is open; passwords are never saved.</CardDescription></CardHeader><CardContent className="space-y-4">
      <LiveConnections connections={connections} source={source} onSource={onSource} disabled={false} />
      {source ? <p className="text-sm">Using <strong>{source.name}</strong> · {Object.keys(source.schemas).length} tables. Open Schema review to inspect the imported structure.</p> : <p className="text-sm text-muted-foreground">No database schema selected. You can still generate downloadable CSV data from a chat prompt.</p>}
      {connections.profiles.length > 0 && <p className="text-xs text-muted-foreground">{connections.profiles.length} connection profile{connections.profiles.length === 1 ? "" : "s"} in this session</p>}
    </CardContent></Card>
  </WorkspaceSection>;
}

export function SchemaWorkspace({ source, scenario }: { source: SchemaSource | null; scenario?: ScenarioConfiguration }) {
  const [selection, setSelection] = useState<"source" | "scenario">("scenario");
  const hasScenario = !!scenario?.schemas && Object.keys(scenario.schemas).length > 0;
  const schemas = selection === "source" && source ? source.schemas : hasScenario ? scenario!.schemas! : source?.schemas;
  return <WorkspaceSection title="Schema review" description="Inspect the tables and columns used by the current chat.">
    {(source || hasScenario) && <div className="flex flex-wrap gap-2">
      {hasScenario && <Button variant={selection === "scenario" || !source ? "secondary" : "outline"} onClick={() => setSelection("scenario")}>Scenario schema</Button>}
      {source && <Button variant={selection === "source" || !hasScenario ? "secondary" : "outline"} onClick={() => setSelection("source")}>Imported schema · {source.name}</Button>}
    </div>}
    {schemas ? <Card><CardHeader><CardTitle>{selection === "source" && source ? source.name : hasScenario ? scenario!.title : source?.name}</CardTitle><CardDescription>Structure only · {Object.keys(schemas).length} tables</CardDescription></CardHeader><CardContent><SchemaExplorer schemas={schemas} /></CardContent></Card> : <Card><CardContent className="py-8 text-sm text-muted-foreground">No schema to review yet. Connect a database or describe a scenario in the chat.</CardContent></Card>}
  </WorkspaceSection>;
}

export function RunHistoryWorkspace() {
  const [jobs, setJobs] = useState<GenerationJob[]>([]);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const response = await fetch("/api/jobs?limit=100");
        if (!response.ok) throw new Error(`Could not load runs (${response.status}).`);
        const data = await response.json() as GenerationJob[];
        if (active) { setJobs(data); setError(""); }
      } catch (cause) {
        if (active) setError(cause instanceof Error ? cause.message : "Run history is unavailable.");
      } finally { if (active) setLoading(false); }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 5000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);
  const visible = jobs.filter(job => (filter === "all" || job.status === filter) && `${job.scenario?.title ?? ""} ${job.jobId}`.toLowerCase().includes(query.toLowerCase()));
  return <WorkspaceSection title="Run history" description="Inspect live generation progress, outputs, and evaluation.">
    {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
    {loading && !jobs.length ? <p className="flex items-center gap-2 text-sm text-muted-foreground"><Spinner /> Loading runs…</p> : !jobs.length ? <Card><CardContent className="py-8 text-sm text-muted-foreground">No generation runs yet. Review a plan in the chat and generate a dataset to see it here.</CardContent></Card> : <Card><CardHeader><CardTitle>Runs</CardTitle><CardDescription>{jobs.length} most recent · refreshes automatically</CardDescription></CardHeader><CardContent className="space-y-4">
      <div className="flex flex-wrap gap-2"><Input aria-label="Search runs" className="max-w-sm" placeholder="Search scenario or run ID" value={query} onChange={event => setQuery(event.target.value)} /><select aria-label="Filter run status" className="h-9 rounded-xl border bg-background px-2 text-xs" value={filter} onChange={event => setFilter(event.target.value)}><option value="all">All statuses</option>{["generating", "validating", "evaluating", "complete", "failed"].map(status => <option key={status} value={status}>{status}</option>)}</select></div>
      {visible.length ? visible.map(job => <div key={job.jobId} className="flex flex-wrap items-center justify-between gap-3 border-b py-3 last:border-b-0"><div className="min-w-0"><p className="font-medium">{job.scenario?.title ?? `Run ${job.jobId}`}</p><p className="break-all text-xs text-muted-foreground">{job.jobId} · {job.createdAt ? new Date(job.createdAt).toLocaleString() : "Date unavailable"}</p></div><Badge variant={job.status === "failed" ? "destructive" : job.status === "complete" ? "secondary" : "outline"}>{job.status} · {job.progress}%</Badge><Link className={buttonVariants({ size: "sm", variant: "outline" })} to={`/runs/${job.jobId}`}>View run</Link></div>) : <p className="py-4 text-sm text-muted-foreground">No runs match those filters.</p>}
    </CardContent></Card>}
  </WorkspaceSection>;
}

function WorkspaceSection({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  return <main className="min-h-0 flex-1 overflow-y-auto px-5 py-8 sm:px-8"><div className="mx-auto flex max-w-3xl flex-col gap-5"><div><h1 className="text-2xl font-semibold tracking-tight">{title}</h1><p className="mt-1 text-sm text-muted-foreground">{description}</p></div>{children}</div></main>;
}

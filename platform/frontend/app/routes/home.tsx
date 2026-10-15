import { LiveConnections, useLiveProfiles, type SchemaSource } from "~/components/studio/live-connections";
import { SchemaExplorer } from "~/components/studio/schema-explorer";
import { RunQualitySummary } from "~/components/studio/run-quality-summary";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import { DefaultChatTransport, type UIMessage } from "ai";
import { useChat } from "@ai-sdk/react";
import { Link, useLocation, useNavigate, useParams } from "react-router";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  AiMagicIcon,
  ArrowUp02Icon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";

import type { Route } from "./+types/home";
import { Button, buttonVariants } from "~/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "~/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "~/components/ui/alert";
import { Badge } from "~/components/ui/badge";
import {
  Bubble,
  BubbleContent,
} from "~/components/ui/bubble";
import {
  Message,
  MessageAvatar,
  MessageContent,
} from "~/components/ui/message";
import {
  MessageScroller,
  MessageScrollerButton,
  MessageScrollerContent,
  MessageScrollerItem,
  MessageScrollerProvider,
  MessageScrollerViewport,
} from "~/components/ui/message-scroller";
import { Spinner } from "~/components/ui/spinner";
import { Textarea } from "~/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { Progress, ProgressLabel, ProgressValue } from "~/components/ui/progress";
import { AppShell } from "~/components/studio/app-shell";
import { DataPreview } from "~/components/studio/data-preview";
import { ConnectionsWorkspace, RunHistoryWorkspace, SchemaWorkspace } from "~/components/studio/chat-workspace";
import type { ChatRunReference, CostEstimate, GenerationJob, SavedConversation, ScenarioConfiguration } from "~/lib/studio-types";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Syda — Synthetic data, shaped by you" },
    {
      name: "description",
      content: "Design, generate, and evaluate trustworthy synthetic data.",
    },
  ];
}

const promptSuggestions = [
  {
    label: "Insurance claims",
    prompt:
      "Design a synthetic data scenario for 10,000 insurance claims where 15% are denied, with realistic diagnoses, providers, and payment timelines.",
  },
  {
    label: "Patient journeys",
    prompt:
      "Design a scenario for 5,000 longitudinal patient journeys with diagnoses, treatments, follow-ups, and realistic care gaps.",
  },
  {
    label: "E-commerce orders",
    prompt:
      "Plan an e-commerce dataset with 25,000 orders, customers, products, refunds, and seasonal purchasing patterns.",
  },
];

type SydaMessage = UIMessage<unknown, { scenario: ScenarioConfiguration }>;

type ChatRun = ChatRunReference;

function titleForPrompt(prompt: string) {
  const normalized = prompt.replace(/\s+/g, " ").trim();
  if (/^(hi|hello|hey|thanks|thank you|ok|okay|test)\W*$/i.test(normalized)) return "New conversation";
  return normalized.length > 72 ? `${normalized.slice(0, 69).trimEnd()}…` : normalized || "New conversation";
}

function isUntitledConversation(title: string) {
  return !title || title === "New conversation" || /^(hi|hello|hey|thanks|thank you|ok|okay|test)\W*$/i.test(title);
}

function firstUserPrompt(messages: SydaMessage[]) {
  const firstUser = messages.find((message) => message.role === "user");
  return firstUser?.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join(" ") ?? "New conversation";
}

let legacyChatMigration: Promise<SavedConversation> | undefined;

function createMigratedConversation(title: string) {
  legacyChatMigration ??= fetch("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  }).then(async (response) => {
    if (!response.ok) throw new Error(`Could not save the previous chat (${response.status}).`);
    return await response.json() as SavedConversation;
  }).finally(() => { legacyChatMigration = undefined; });
  return legacyChatMigration;
}

function PromptComposer({
  value,
  onChange,
  onSubmit,
  compact = false,
  disabled = false,
  isLoading = false,
  source, onSource, connections,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  compact?: boolean;
  disabled?: boolean;
  isLoading?: boolean;
  connections: ReturnType<typeof useLiveProfiles>;
  source: SchemaSource | null;
  onSource: (source: SchemaSource | null) => void;
}) {
  const [attachmentError, setAttachmentError] = useState("");
  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !disabled && !isLoading) {
      event.preventDefault();
      onSubmit();
    }
  };

  return (
    <form
      className="w-full"
      onSubmit={(event: FormEvent) => {
        event.preventDefault();
        if (!disabled && !isLoading) {
          onSubmit();
        }
      }}
    >
      <div className="relative overflow-hidden rounded-3xl border bg-card shadow-[0_14px_44px_-28px_oklch(0.2_0.02_260/0.45)] focus-within:border-ring focus-within:ring-1 focus-within:ring-ring/30">
        <Textarea
          aria-label="Describe your synthetic data scenario"
          autoFocus={!compact}
          className={compact ? "min-h-20 resize-none rounded-none border-0 shadow-none focus-visible:ring-0" : "min-h-28 resize-none rounded-none border-0 shadow-none focus-visible:ring-0"}
          disabled={disabled || isLoading}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={isLoading ? "Syda Agent is analyzing and compiling your scenario…" : "Describe the data you want to generate…"}
          value={value}
        />
        <div className="flex flex-wrap items-center gap-2 border-t px-3 py-3">
          <label className="inline-flex h-9 shrink-0 cursor-pointer items-center justify-center rounded-full border px-3 text-sm font-medium focus-within:ring-2 focus-within:ring-ring">
            Attach<input className="sr-only" aria-label="Attach schema or sample" disabled={disabled || isLoading} type="file" accept=".csv,.json,.txt" onChange={async e => { const file = e.target.files?.[0]; e.target.value = ""; if (!file) return; setAttachmentError(""); if (file.size > 100000 || !/\.(csv|json|txt)$/i.test(file.name)) { setAttachmentError("Choose a CSV, JSON, or TXT file under 100 KB."); return; } try { onChange(value + `\n\nAttached context (${file.name}):\n${await file.text()}`); } catch { setAttachmentError("Could not read this file. Please retry."); } }} />
          </label>
          <LiveConnections connections={connections} source={source} onSource={onSource} disabled={disabled || isLoading} />
          <Button aria-label="Send prompt" className="ml-auto shrink-0" disabled={!value.trim() || disabled || isLoading} size="icon" type="submit">
          {isLoading ? (
            <Spinner />
          ) : (
            <HugeiconsIcon icon={ArrowUp02Icon} strokeWidth={2} />
          )}
        </Button>
        </div>
      </div>
      {attachmentError && <p role="alert" className="mt-2 text-xs text-destructive">{attachmentError}</p>}
      {!compact && (
        <p className="mt-3 text-center text-xs text-muted-foreground">
          Syda can make mistakes. Review rules and evaluations before using generated data.
        </p>
      )}
    </form>
  );
}

function ScenarioCard({
  scenario,
  onEdit,
  onGenerate,
  messageId,
  sourceName,
}: {
  scenario: ScenarioConfiguration;
  onEdit: (scenario: ScenarioConfiguration, messageId: string) => Promise<void>;
  onGenerate: (scenario: ScenarioConfiguration) => Promise<void>;
  messageId: string;
  sourceName: string;
}) {
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string>();
  const [reviewOpen, setReviewOpen] = useState(false);
  const [estimate, setEstimate] = useState<CostEstimate>();
  const [estimateLoading, setEstimateLoading] = useState(false);
  const [estimateError, setEstimateError] = useState<string>();
  const [generationError, setGenerationError] = useState<string>();

  const openEditor = async () => {
    setSaving(true);
    setSaveError(undefined);
    try {
      await onEdit(scenario, messageId);
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : "Could not save the scenario.");
    } finally {
      setSaving(false);
    }
  };

  const openReview = async () => {
    setReviewOpen(true);
    setEstimate(undefined);
    setEstimateError(undefined);
    setGenerationError(undefined);
    setEstimateLoading(true);
    try {
      const response = await fetch("/api/estimate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => undefined) as { detail?: string } | undefined;
        throw new Error(typeof body?.detail === "string" ? body.detail : `Could not estimate this run (${response.status}).`);
      }
      setEstimate(await response.json() as CostEstimate);
    } catch (error) {
      setEstimateError(error instanceof Error ? error.message : "Could not estimate this run.");
    } finally {
      setEstimateLoading(false);
    }
  };

  const startGeneration = async () => {
    setSaving(true);
    setGenerationError(undefined);
    try {
      await onGenerate(scenario);
      setReviewOpen(false);
    } catch (error) {
      setGenerationError(error instanceof Error ? error.message : "Could not start dataset generation.");
    } finally {
      setSaving(false);
    }
  };

  const tableEntries = Object.entries(scenario.schemas ?? {});
  const totalPathWeight = scenario.paths?.reduce((total, path) => total + path.weight, 0) ?? 0;

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <CardTitle>{scenario.title}</CardTitle>
              <CardDescription>{scenario.description}</CardDescription>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge variant="secondary">Draft plan</Badge>
              <Badge variant="outline">{scenario.recordCount.toLocaleString()} instances</Badge>
              <Badge variant="outline">{Object.keys(scenario.schemas ?? {}).length} tables</Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          {!!scenario.workflow.length && <section className="flex flex-col gap-2" aria-labelledby="scenario-workflow">
            <h3 className="text-xs font-medium" id="scenario-workflow">Workflow</h3>
            <div className="flex flex-wrap gap-1.5">
              {scenario.workflow.map((step, index) => (
                <Badge key={`${step}-${index}`} variant="secondary">{index + 1}. {step}</Badge>
              ))}
            </div>
          </section>}
          {!!scenario.paths?.length && (
            <section className="flex flex-col gap-2" aria-labelledby="scenario-paths">
              <h3 className="text-xs font-medium" id="scenario-paths">Scenario paths</h3>
              <div className="flex flex-wrap gap-1.5">
                {scenario.paths.map((path) => (
                  <Badge key={path.name} variant="outline">
                    {path.name} · {Math.round((path.weight / totalPathWeight) * 100)}%
                  </Badge>
                ))}
              </div>
            </section>
          )}
          {!!tableEntries.length && <section className="flex flex-col gap-2" aria-labelledby="scenario-schema">
            <h3 className="text-xs font-medium" id="scenario-schema">Schema preview</h3>
            <details><summary className="cursor-pointer text-xs">Browse {tableEntries.length} tables and columns</summary><div className="mt-3"><SchemaExplorer schemas={scenario.schemas ?? {}} /></div></details>
          </section>}
          {!!scenario.rules.length && <section className="flex flex-col gap-2" aria-labelledby="scenario-rules">
            <h3 className="text-xs font-medium" id="scenario-rules">Rules</h3>
            <ul className="list-disc pl-4 text-xs text-muted-foreground">
              {scenario.rules.map((rule, index) => <li key={`${rule}-${index}`}>{rule}</li>)}
            </ul>

          </section>}
        </CardContent>
        <CardFooter className="flex-wrap justify-between gap-2">
          <Button disabled={saving} onClick={() => void openEditor()} type="button" variant="outline">
            {saving ? <Spinner data-icon="inline-start" /> : null}{saving ? "Opening editor…" : "Edit full scenario"}
          </Button>
          <Button onClick={() => void openReview()} type="button">
            <HugeiconsIcon data-icon="inline-start" icon={SparklesIcon} strokeWidth={2} /> Review plan
          </Button>
        </CardFooter>
      </Card>

      {saveError && <Alert variant="destructive"><AlertTitle>Could not open editor</AlertTitle><AlertDescription>{saveError}</AlertDescription></Alert>}

      <Dialog onOpenChange={setReviewOpen} open={reviewOpen}>
        <DialogContent className="max-h-[calc(100svh-2rem)] max-w-lg overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Review before generating</DialogTitle>
            <DialogDescription>This scenario is a plan. No dataset has been generated yet.</DialogDescription>
          </DialogHeader>
          <div className="space-y-5">
            <div>
              <h3 className="font-medium">{scenario.title}</h3>
              <p className="mt-1 text-xs leading-5 text-muted-foreground">{scenario.description}</p>
            </div>
            <dl className="grid grid-cols-2 gap-3 rounded-2xl border bg-muted/30 p-4 text-xs">
              <div><dt className="text-muted-foreground">Scenario instances</dt><dd className="mt-1 font-medium tabular-nums">{scenario.recordCount.toLocaleString()}</dd></div>
              <div><dt className="text-muted-foreground">Connected tables</dt><dd className="mt-1 font-medium tabular-nums">{tableEntries.length || scenario.workflow.length}</dd></div>
              <div><dt className="text-muted-foreground">Workflow steps</dt><dd className="mt-1 font-medium tabular-nums">{scenario.workflow.length}</dd></div>
              <div><dt className="text-muted-foreground">Rules</dt><dd className="mt-1 font-medium tabular-nums">{scenario.rules.length}</dd></div>
            </dl>
            <section className="space-y-2">
              <h3 className="text-xs font-medium">Workflow</h3>
              <p className="text-xs leading-5 text-muted-foreground">{scenario.workflow.join(" → ")}</p>
            </section>
            <section className="space-y-2">
              <h3 className="text-xs font-medium">Generation rules</h3>
              <ul className="list-disc space-y-1 pl-4 text-xs text-muted-foreground">
                {scenario.rules.map((rule, index) => <li key={`${rule}-${index}`}>{rule}</li>)}
              </ul>

            </section>
            <section className="rounded-2xl border p-4 text-xs space-y-2"><h3 className="font-medium">Source and destination</h3><p>Source: {sourceName}. Review the {tableEntries.length} table definitions below before generating.</p><p>Destination: downloadable CSV files. No database write-back.</p><details><summary className="cursor-pointer">Review table definitions</summary><div className="mt-3"><SchemaExplorer schemas={scenario.schemas ?? {}} /></div></details></section>
            <section className="rounded-2xl border p-4">
              <div className="flex items-start justify-between gap-3">
                <div><h3 className="text-xs font-medium">Generation estimate</h3><p className="mt-1 text-[10px] text-muted-foreground">{estimate ? `${estimate.provider} · ${estimate.model}` : "Checking configured provider…"}</p></div>
                <p className="text-sm font-semibold tabular-nums">{estimateLoading ? "…" : estimate?.estimatedCostUsd === 0 ? "No model cost" : estimate?.estimatedCostUsd == null ? "Unavailable" : `~$${estimate.estimatedCostUsd.toFixed(4)}`}</p>
              </div>
              {(estimate?.note || estimateError) && <p className="mt-2 text-[10px] leading-4 text-muted-foreground">{estimateError ?? estimate?.note}</p>}
            </section>
            {generationError && <Alert variant="destructive"><AlertTitle>Could not start generation</AlertTitle><AlertDescription>{generationError}</AlertDescription></Alert>}
          </div>
          <DialogFooter className="sm:justify-between">
            <Button disabled={saving} onClick={() => void openEditor()} type="button" variant="outline">Edit full scenario</Button>
            <Button disabled={saving} onClick={() => void startGeneration()} type="button">
              {saving ? <Spinner data-icon="inline-start" /> : null}{saving ? "Starting dataset…" : "Generate dataset"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

    </div>
  );
}

function GenerationRunCard({ run }: { run: ChatRun }) {
  const [job, setJob] = useState<GenerationJob>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const refresh = async () => {
      try {
        const response = await fetch(`/api/jobs/${run.jobId}`);
        if (!response.ok) throw new Error(`Could not load run status (${response.status}).`);
        const nextJob = await response.json() as GenerationJob;
        if (cancelled) return;
        setJob(nextJob);
        setError(undefined);
        if (!(["complete", "failed"].includes(nextJob.status))) {
          timer = window.setTimeout(() => void refresh(), 3000);
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError(fetchError instanceof Error ? fetchError.message : "Run status is unavailable.");
          timer = window.setTimeout(() => void refresh(), 5000);
        }
      }
    };
    void refresh();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [run.jobId]);

  const status = job?.status ?? "generating";
  const complete = status === "complete";
  const failed = status === "failed";

  return (
    <Card className="border-primary/20">
      <CardHeader className="flex flex-row items-start justify-between gap-3">
        <div><CardTitle>{complete ? "Dataset ready" : failed ? "Generation failed" : "Generating dataset"}</CardTitle><CardDescription>{run.scenario.title}</CardDescription></div>
        <Badge variant={failed ? "destructive" : complete ? "secondary" : "outline"}>{status}</Badge>
      </CardHeader>
      <CardContent className="space-y-3">
        {!complete && !failed && <Progress value={job?.progress ?? 10}><ProgressLabel>{job?.currentStage ?? "Starting generation worker…"}</ProgressLabel><ProgressValue>{(value) => value}</ProgressValue></Progress>}
        {(error || (failed && job?.currentStage)) && <p className="text-xs text-destructive">{error ?? job?.currentStage}</p>}
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-[10px] text-muted-foreground">Run {run.jobId}</p>
          <div className="flex flex-wrap gap-2">
            {complete && (job?.downloadUrl || job?.filesAvailable) && <a className={buttonVariants({ size: "sm" })} href={job?.downloadUrl ?? `/api/download/${run.jobId}`}>Download dataset</a>}
            <Link className={buttonVariants({ size: "sm", variant: "outline" })} to={`/runs/${run.jobId}?tab=data`}>Browse full dataset</Link>
          </div>
        </div>
        {complete && <RunQualitySummary stats={job?.stats} scenario={run.scenario} />}
        {failed && <Link className={buttonVariants({ size: "sm", variant: "outline" })} to={`/scenarios/${run.scenarioId}`}>Review scenario and retry</Link>}
        {complete && job?.filesAvailable && <DataPreview compact complete jobId={run.jobId} showDownloads={false} />}
      </CardContent>
    </Card>
  );
}

export default function Home() {
  const navigate = useNavigate();
  const location = useLocation();
  const { conversationId } = useParams();
  const conversationIdRef = useRef<string | undefined>(conversationId);
  conversationIdRef.current = conversationId;
  const [prompt, setPrompt] = useState("");
  const [source, setSource] = useState<SchemaSource | null>(null);
  const requestedView = new URLSearchParams(location.search).get("view");
  const workspaceView = requestedView === "connections" || requestedView === "schema" || requestedView === "runs" ? requestedView : "scenario";
  const connections = useLiveProfiles();
  const [chatRuns, setChatRuns] = useState<ChatRun[]>([]);
  const [conversationTitle, setConversationTitle] = useState("");
  const [loadedConversationId, setLoadedConversationId] = useState<string>();
  const [conversationLoading, setConversationLoading] = useState(false);
  const [conversationError, setConversationError] = useState<string>();
  const skipConversationLoadRef = useRef<string | undefined>(undefined);
  const editorReturnRef = useRef<{
    editedScenario?: ScenarioConfiguration;
    messageId: string;
    generatedRun?: ChatRun;
  } | undefined>(undefined);
  const persistedSnapshotRef = useRef("");
  const saveQueueRef = useRef<Promise<void>>(Promise.resolve());
  const chatTransport = useMemo(() => new DefaultChatTransport<SydaMessage>({
    api: "/api/agent/chat",
    prepareSendMessagesRequest: ({ body, id, messages, trigger, messageId }) => ({
      body: {
        ...body,
        id,
        messages,
        trigger,
        messageId,
        conversationId: conversationIdRef.current,
      },
    }),
  }), []);
  const { clearError, error, messages, sendMessage, setMessages, status, stop } = useChat<SydaMessage>({
    transport: chatTransport,
  });
  const isThinking = status === "submitted" || status === "streaming";

  useEffect(() => {
    const returnState = location.state as {
      schemaSource?: SchemaSource;
      editedScenario?: ScenarioConfiguration;
      messageId?: string;
      generatedRun?: ChatRun;
    } | null;
    if (returnState?.schemaSource) {
      setSource(returnState.schemaSource);
      navigate(`${location.pathname}${location.search}`, { replace: true, state: null });
      return;
    }
    if (!returnState?.messageId || (!returnState.editedScenario && !returnState.generatedRun)) return;
    editorReturnRef.current = { ...returnState, messageId: returnState.messageId };
    navigate(`${location.pathname}${location.search}`, { replace: true, state: null });
  }, [location.key, location.pathname, location.search, location.state, navigate]);

  useEffect(() => {
    const returnState = editorReturnRef.current;
    if (!returnState || (conversationId && loadedConversationId !== conversationId)) return;
    if (returnState.editedScenario) {
      setMessages((currentMessages) => currentMessages.map((message) => message.id === returnState.messageId
        ? {
            ...message,
            parts: message.parts.map((part) => part.type === "data-scenario"
              ? { ...part, data: returnState.editedScenario! }
              : part),
          }
        : message));
    }
    if (returnState.generatedRun) {
      setChatRuns((runs) => runs.some((run) => run.jobId === returnState.generatedRun!.jobId)
        ? runs
        : [...runs, returnState.generatedRun!]);
    }
    editorReturnRef.current = undefined;
  }, [conversationId, loadedConversationId, setMessages]);

  const latestScenario = messages
    .flatMap((message) => message.parts)
    .filter((part) => part.type === "data-scenario")
    .at(-1)?.data;

  useEffect(() => {
    let active = true;
    if (conversationId && skipConversationLoadRef.current === conversationId) {
      skipConversationLoadRef.current = undefined;
      return () => { active = false; };
    }

    stop();
    setConversationError(undefined);
    if (!conversationId) {
      setConversationLoading(false);
      setConversationTitle("");
      setLoadedConversationId(undefined);
      conversationIdRef.current = undefined;
      persistedSnapshotRef.current = "";

      // Bring forward the last tab-local transcript once so the new durable
      // history feature does not discard an in-progress pre-history chat.
      let legacyMessages: SydaMessage[] = [];
      let legacyRuns: ChatRun[] = [];
      try {
        const parsedMessages = JSON.parse(sessionStorage.getItem("syda:chat-messages") ?? "[]");
        const parsedRuns = JSON.parse(sessionStorage.getItem("syda:chat-runs") ?? "[]");
        if (Array.isArray(parsedMessages)) legacyMessages = parsedMessages as SydaMessage[];
        if (Array.isArray(parsedRuns)) legacyRuns = parsedRuns as ChatRun[];
      } catch {
        sessionStorage.removeItem("syda:chat-messages");
        sessionStorage.removeItem("syda:chat-runs");
      }

      if (legacyMessages.length && !sessionStorage.getItem("syda:legacy-chat-migrated")) {
        const title = titleForPrompt(firstUserPrompt(legacyMessages));
        void createMigratedConversation(title).then((conversation) => {
          if (!active) return;
          skipConversationLoadRef.current = conversation.id;
          setConversationTitle(conversation.title);
          setChatRuns(legacyRuns);
          setMessages(legacyMessages);
          setLoadedConversationId(conversation.id);
          conversationIdRef.current = conversation.id;
          persistedSnapshotRef.current = JSON.stringify({ title: conversation.title, messages: [], runs: [], scenarioDraft: undefined });
          sessionStorage.setItem("syda:legacy-chat-migrated", "true");
          sessionStorage.removeItem("syda:chat-messages");
          sessionStorage.removeItem("syda:chat-runs");
          window.dispatchEvent(new Event("syda:conversations-changed"));
          navigate(`/app/${conversation.id}`, { replace: true });
        }).catch((migrationError) => {
          if (active) setConversationError(migrationError instanceof Error ? migrationError.message : "Could not migrate the previous chat.");
        });
      } else {
        setMessages([]);
        setChatRuns([]);
      }
      return () => { active = false; };
    }

    setConversationLoading(true);
    setLoadedConversationId(undefined);
    void fetch(`/api/conversations/${conversationId}`)
      .then(async (response) => {
        if (!response.ok) throw new Error(response.status === 404 ? "Chat not found." : `Could not load chat (${response.status}).`);
        return await response.json() as SavedConversation;
      })
      .then((conversation) => {
        if (!active) return;
        const restoredMessages = [...conversation.messages] as SydaMessage[];
        if (conversation.scenarioDraft) {
          let assistantIndex = -1;
          for (let index = restoredMessages.length - 1; index >= 0; index -= 1) {
            if (restoredMessages[index].role === "assistant") {
              assistantIndex = index;
              break;
            }
          }
          const assistantMessage = assistantIndex >= 0
            ? restoredMessages[assistantIndex]
            : { id: crypto.randomUUID(), role: "assistant" as const, parts: [] };
          const scenarioPart = { type: "data-scenario" as const, id: "scenario", data: conversation.scenarioDraft };
          const parts = assistantMessage.parts.filter((part) => part.type !== "data-scenario") as SydaMessage["parts"];
          parts.push(scenarioPart as SydaMessage["parts"][number]);
          const restoredAssistant = { ...assistantMessage, parts } as SydaMessage;
          if (assistantIndex >= 0) restoredMessages[assistantIndex] = restoredAssistant;
          else restoredMessages.push(restoredAssistant);
        }
        const restoredRuns = conversation.runs ?? [];
        setConversationTitle(conversation.title);
        setMessages(restoredMessages);
        setChatRuns(restoredRuns);
        setLoadedConversationId(conversation.id);
        conversationIdRef.current = conversation.id;
        persistedSnapshotRef.current = JSON.stringify({
          title: conversation.title,
          messages: restoredMessages,
          runs: restoredRuns,
          scenarioDraft: conversation.scenarioDraft ?? undefined,
        });
      })
      .catch((loadError) => {
        if (active) setConversationError(loadError instanceof Error ? loadError.message : "Could not load chat history.");
      })
      .finally(() => { if (active) setConversationLoading(false); });
    return () => { active = false; };
  }, [conversationId, navigate, setMessages, stop]);

  useEffect(() => {
    if (!conversationId || loadedConversationId !== conversationId || status === "streaming") return;
    const payload = { title: conversationTitle, messages, runs: chatRuns, scenarioDraft: latestScenario };
    const snapshot = JSON.stringify(payload);
    if (snapshot === persistedSnapshotRef.current) return;
    saveQueueRef.current = saveQueueRef.current
      .catch(() => {})
      .then(async () => {
        const response = await fetch(`/api/conversations/${conversationId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: snapshot,
        });
        if (!response.ok) throw new Error(`Could not save chat history (${response.status}).`);
        persistedSnapshotRef.current = snapshot;
        setConversationError(undefined);
        window.dispatchEvent(new Event("syda:conversations-changed"));
      })
      .catch((saveError) => {
        setConversationError(saveError instanceof Error ? saveError.message : "Could not save chat history.");
      });
  }, [chatRuns, conversationId, conversationTitle, latestScenario, loadedConversationId, messages, status]);

  useEffect(() => () => {
    stop();
  }, [stop]);

  const submitPrompt = async (promptOverride?: string) => {
    const text = (typeof promptOverride === "string" ? promptOverride : prompt).trim();
    if (!text || isThinking || conversationLoading || (conversationId && loadedConversationId !== conversationId)) return;

    setConversationError(undefined);
    if (!conversationId) {
      try {
        const response = await fetch("/api/conversations", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: titleForPrompt(text) }),
        });
        if (!response.ok) throw new Error(`Could not save this chat (${response.status}).`);
        const conversation = await response.json() as SavedConversation;
        skipConversationLoadRef.current = conversation.id;
        conversationIdRef.current = conversation.id;
        setConversationTitle(conversation.title);
        setChatRuns([]);
        setLoadedConversationId(conversation.id);
        persistedSnapshotRef.current = JSON.stringify({ title: conversation.title, messages: [], runs: [], scenarioDraft: undefined });
        window.dispatchEvent(new Event("syda:conversations-changed"));
        navigate(`/app/${conversation.id}`, { replace: true });
      } catch (createError) {
        setConversationError(createError instanceof Error ? createError.message : "Could not save this chat.");
        return;
      }
    }

    if (conversationId && isUntitledConversation(conversationTitle) && !isUntitledConversation(text)) {
      setConversationTitle(titleForPrompt(text));
    }

    setPrompt("");
    clearError();
    await sendMessage({ text: source ? `${text}\n\nDatabase schema context (${source.name}; structure only, output CSV):\n${JSON.stringify(source.schemas)}` : text });
  };

  const startNewScenario = () => {
    setSource(null);
    stop();
    setPrompt("");
    setMessages([]);
    setChatRuns([]);
    setConversationTitle("");
    setLoadedConversationId(undefined);
    setConversationError(undefined);
    conversationIdRef.current = undefined;
    persistedSnapshotRef.current = "";
    navigate("/app");
  };

  const openEditor = async (scenario: ScenarioConfiguration, messageId: string) => {
    const response = await fetch("/api/scenarios", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(scenario),
    });
    if (!response.ok) throw new Error(`Could not save scenario (${response.status}).`);
    const saved = await response.json() as { id: string };
    window.dispatchEvent(new Event("syda:scenarios-changed"));
    const returnTo = conversationId ? `/app/${conversationId}` : "/app";
    navigate(`/scenarios/${saved.id}?returnTo=${encodeURIComponent(returnTo)}&messageId=${encodeURIComponent(messageId)}`);
  };

  const startGeneration = async (scenario: ScenarioConfiguration, messageId: string) => {
    const validationResponse = await fetch("/api/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario }),
    });
    if (!validationResponse.ok) throw new Error(`Could not validate this scenario (${validationResponse.status}).`);
    const validation = await validationResponse.json() as { valid: boolean; schema_errors?: string[] };
    if (!validation.valid) throw new Error(validation.schema_errors?.join(" ") || "Fix the scenario before generating.");

    const saveResponse = await fetch("/api/scenarios", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(scenario),
    });
    if (!saveResponse.ok) throw new Error(`Could not save this scenario (${saveResponse.status}).`);
    const saved = await saveResponse.json() as { id: string };
    window.dispatchEvent(new Event("syda:scenarios-changed"));

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario, scenarioId: saved.id }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => undefined) as { detail?: string } | undefined;
        throw new Error(body?.detail ?? `Could not start generation (${response.status}).`);
      }
      const job = await response.json() as { jobId: string };
      setChatRuns((runs) => [...runs, { messageId, jobId: job.jobId, scenarioId: saved.id, scenario }]);
    } catch (error) {
      throw new Error(`The scenario was saved, but generation could not start. ${error instanceof Error ? error.message : "Try again from the scenario editor."}`);
    }
  };

  return (
    <AppShell title={conversationTitle || latestScenario?.title || "New scenario"} subtitle={messages.length ? "Scenario design chat" : "Describe what you want to generate"} onNewScenario={startNewScenario}>

        {conversationError && <div className="mx-auto mt-4 w-full max-w-3xl px-5"><Alert variant="destructive"><AlertTitle>Chat history issue</AlertTitle><AlertDescription>{conversationError}</AlertDescription></Alert></div>}

        {workspaceView === "connections" ? <ConnectionsWorkspace connections={connections} source={source} onSource={setSource} /> : workspaceView === "schema" ? <SchemaWorkspace source={source} scenario={latestScenario} /> : workspaceView === "runs" ? <RunHistoryWorkspace /> : conversationLoading ? (
          <main className="flex min-h-0 flex-1 items-center justify-center gap-2 text-sm text-muted-foreground"><Spinner /> Loading chat history…</main>
        ) : !messages.length ? (
          <main className="relative flex min-h-0 flex-1 items-center justify-center overflow-y-auto px-5 py-12">
            <div aria-hidden="true" className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_38%,oklch(0.96_0.025_265),transparent_38%)]" />
            <div className="relative flex w-full max-w-2xl flex-col items-center">
              <div className="mb-5 flex size-10 items-center justify-center border bg-card shadow-sm">
                <HugeiconsIcon icon={SparklesIcon} strokeWidth={1.8} />
              </div>
              <p className="mb-2 text-xs font-medium text-muted-foreground">New synthetic data scenario</p>
              <h1 className="max-w-lg text-center text-3xl font-semibold tracking-[-0.035em] sm:text-4xl">What should we generate?</h1>
              <p className="mb-8 mt-3 max-w-md text-center text-sm leading-6 text-muted-foreground">
                Describe the entities, relationships, constraints, and edge cases. Syda will turn them into a testable scenario.
              </p>
              <PromptComposer connections={connections} source={source} onSource={setSource} disabled={isThinking || conversationLoading || (!!conversationId && loadedConversationId !== conversationId)} isLoading={isThinking} onChange={setPrompt} onSubmit={submitPrompt} value={prompt} />
              <div className="mt-5 flex flex-wrap justify-center gap-2">
                {promptSuggestions.map((suggestion) => (
                  <Button key={suggestion.label} onClick={() => setPrompt(suggestion.prompt)} size="sm" type="button" variant="outline">
                    {suggestion.label}
                  </Button>
                ))}
              </div>
            </div>
          </main>
        ) : (
          <main className="flex min-h-0 flex-1 flex-col">
            <MessageScrollerProvider autoScroll>
              <MessageScroller className="flex-1">
                <MessageScrollerViewport>
                  <MessageScrollerContent className="mx-auto w-full max-w-3xl px-5 py-8 sm:px-8 sm:py-10">
                    {messages.map((message, index) => {
                      const text = message.parts
                        .filter((part) => part.type === "text")
                        .map((part) => part.text)
                        .join("");
                      const scenarioPart = message.parts.find((part) => part.type === "data-scenario");
                      const candidateScenario = scenarioPart?.data;
                      const scenario = candidateScenario
                        && candidateScenario.workflow?.length >= 2
                        && candidateScenario.rules?.length > 0
                        && Object.keys(candidateScenario.schemas ?? {}).length > 0
                        ? candidateScenario
                        : undefined;
                      const isStreamingThis = status === "streaming" && index === messages.length - 1 && message.role === "assistant";

                      return (
                        <MessageScrollerItem
                          key={message.id}
                          messageId={message.id}
                          scrollAnchor={message.role === "user"}
                        >
                          <Message align={message.role === "user" ? "end" : "start"}>
                            {message.role === "assistant" && (
                              <MessageAvatar className="self-start bg-primary text-primary-foreground">
                                <HugeiconsIcon icon={AiMagicIcon} strokeWidth={2} />
                              </MessageAvatar>
                            )}
                            <MessageContent>
                              <Bubble align={message.role === "user" ? "end" : "start"} className={scenario ? "w-full" : undefined} variant={message.role === "user" ? "secondary" : "ghost"}>
                                <BubbleContent className={scenario ? "w-full" : undefined}>
                                  {text && (
                                    <p className={scenario ? "mb-3 text-xs leading-5" : undefined}>
                                      {text.split("\n\nDatabase schema context (")[0]}
                                      {text.includes("\n\nDatabase schema context (") && <span className="mt-2 block text-xs text-muted-foreground">Database schema included with this prompt.</span>}
                                      {isStreamingThis && (
                                        <span className="ml-1 inline-block h-3.5 w-1.5 animate-pulse rounded-xs bg-primary align-middle" />
                                      )}
                                    </p>
                                  )}
                                  {scenario && (
                                    <ScenarioCard
                                      sourceName={messages.slice(0, index).reverse().find(item => item.role === "user")?.parts.filter(part => part.type === "text").map(part => part.text).join(" ").match(/Database schema context \((.*); structure only, output CSV\):/)?.[1] ?? "Chat instructions"}
                                      messageId={message.id}
                                      onEdit={openEditor}
                                      onGenerate={(nextScenario) => startGeneration(nextScenario, message.id)}
                                      scenario={scenario}
                                    />
                                  )}
                                  {chatRuns.filter((run) => run.messageId === message.id).map((run) => <GenerationRunCard key={run.jobId} run={run} />)}
                                </BubbleContent>
                              </Bubble>
                            </MessageContent>
                          </Message>
                        </MessageScrollerItem>
                      );
                    })}
                    {status === "submitted" && (
                      <MessageScrollerItem messageId="thinking-indicator" scrollAnchor>
                          <Message align="start">
                          <MessageAvatar className="self-start bg-primary text-primary-foreground">
                            <HugeiconsIcon icon={AiMagicIcon} strokeWidth={2} />
                          </MessageAvatar>
                          <MessageContent>
                            <Bubble align="start" className="w-full" variant="ghost">
                              <BubbleContent className="w-full">
                                <div className="flex items-center gap-2 py-1 text-xs text-muted-foreground">
                                  <Spinner />
                                  <span className="shimmer">Syda Agent is analyzing schema & compiling scenario rules…</span>
                                </div>
                              </BubbleContent>
                            </Bubble>
                          </MessageContent>
                        </Message>
                      </MessageScrollerItem>
                    )}
                  </MessageScrollerContent>
                </MessageScrollerViewport>
                <MessageScrollerButton />
              </MessageScroller>
            </MessageScrollerProvider>
            <div className="shrink-0 border-t bg-background px-5 py-4">
              <div className="mx-auto w-full max-w-3xl">
                <div className="flex items-end gap-2">
                  <PromptComposer connections={connections} source={source} onSource={setSource} compact disabled={isThinking || conversationLoading || (!!conversationId && loadedConversationId !== conversationId)} isLoading={isThinking} onChange={setPrompt} onSubmit={submitPrompt} value={prompt} />
                  {isThinking && (
                    <Button onClick={() => stop()} type="button" variant="outline">
                      Stop
                    </Button>
                  )}
                </div>
                {error && <p className="mt-2 text-xs text-destructive">{error.message}</p>}
              </div>
            </div>
          </main>
        )}
    </AppShell>
  );
}

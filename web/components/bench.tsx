"use client";

import { useEffect, useMemo, useState, type FormEvent } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { PRESETS, QUESTIONS } from "@/lib/presets";
import {
  headLabel,
  formatAnswer,
  type DecideResponse,
  type FieldResult,
  type Health,
  type Plan,
  type Preset,
  type Shape,
} from "@/lib/types";

const API = "/dynajev-api";

const SHAPES: { id: Shape; label: string }[] = [
  { id: "boolean", label: "Yes / no" },
  { id: "categorical", label: "One of" },
  { id: "ordinal", label: "Rating" },
  { id: "multilabel", label: "Flags" },
  { id: "open", label: "Open" },
  { id: "schema", label: "Schema" },
  { id: "questions", label: "Typed questions (JSON)" },
];

export function Bench() {
  const [presetId, setPresetId] = useState(PRESETS[0].id);
  const [shape, setShape] = useState<Shape>(PRESETS[0].shape);
  const [context, setContext] = useState(PRESETS[0].context);
  const [question, setQuestion] = useState(PRESETS[0].question);
  const [options, setOptions] = useState(PRESETS[0].options);
  const [levels, setLevels] = useState(PRESETS[0].levels);
  const [schema, setSchema] = useState(PRESETS[0].schema);
  const [questions, setQuestions] = useState(PRESETS[0].questions ?? QUESTIONS);
  const [strategy, setStrategy] = useState(PRESETS[0].strategy);
  const [examples, setExamples] = useState(PRESETS[0].examples);
  const [health, setHealth] = useState<Health | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DecideResponse | null>(null);
  const [openPrompt, setOpenPrompt] = useState<string | null>(null);

  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const response = await fetch(`${API}/api/health`);
        if (!response.ok) return;
        const body = (await response.json()) as Health;
        if (!stop) setHealth(body);
        if (body.loaded || body.error) return;
      } catch {
        if (!stop) setHealth(null);
      }
      window.setTimeout(tick, 2000);
    };
    void tick();
    return () => {
      stop = true;
    };
  }, []);

  function applyPreset(preset: Preset) {
    setPresetId(preset.id);
    setShape(preset.shape);
    setContext(preset.context);
    setQuestion(preset.question);
    setOptions(preset.options);
    setLevels(preset.levels);
    setSchema(preset.schema);
    setQuestions(preset.questions ?? QUESTIONS);
    setStrategy(preset.strategy);
    setExamples(preset.examples);
    setError(null);
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setRunning(true);
    try {
      const body: Record<string, unknown> = { context, strategy };
      if (shape === "schema") {
        body.schema = JSON.parse(schema);
      } else if (shape === "questions") {
        body.questions = JSON.parse(questions);
      } else {
        body.type = shape;
        body.question = question;
        if (shape === "categorical") body.options = lines(options);
        if (shape === "multilabel") {
          body.options = lines(options);
          body.exclusive = false;
        }
        if (shape === "ordinal") body.levels = lines(levels);
      }
      if (examples?.length) body.examples = examples;
      const response = await fetch(`${API}/api/decide`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = await response.json();
      if (!response.ok) {
        const detail = typeof payload.detail === "string" ? payload.detail : "The readout failed.";
        throw new Error(detail);
      }
      setResult(payload as DecideResponse);
      setOpenPrompt(null);
    } catch (err) {
      setResult(null);
      setError(err instanceof SyntaxError ? "The JSON is not valid." : (err as Error).message);
    } finally {
      setRunning(false);
    }
  }

  const ready = health?.loaded === true;
  const status = useMemo(() => describeHealth(health), [health]);

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-8 sm:px-6 sm:py-12">
      <header className="flex flex-col gap-6 border-b border-rule pb-8 sm:flex-row sm:items-end sm:justify-between">
        <div className="max-w-2xl">
          <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-ink-soft">Dynajev</p>
          <h1 className="mt-3 font-serif text-4xl leading-[1.05] text-foreground sm:text-5xl">
            The head is compiled from the question.
          </h1>
          <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-ink-soft">
            A frozen open model stays put. Yes/no, a choice, a rating, a set of flags, and a schema each get a
            different readout, built at request time from the shape of the answer.
          </p>
        </div>
        <div className="min-w-56 border border-rule bg-paper px-4 py-3">
          <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink-soft">Trunk</p>
          <p className="mt-1 text-sm leading-snug">{status}</p>
        </div>
      </header>

      <div className="flex gap-2 overflow-x-auto pb-1">
        {PRESETS.map((preset) => {
          const active = preset.id === presetId;
          return (
            <button
              key={preset.id}
              type="button"
              onClick={() => applyPreset(preset)}
              className={`min-w-36 shrink-0 border px-3 py-2 text-left ${
                active ? "border-foreground bg-foreground text-paper" : "border-rule bg-paper text-foreground"
              }`}
            >
              <span className="block text-sm">{preset.name}</span>
              <span className={`mt-0.5 block font-mono text-[11px] ${active ? "text-paper/70" : "text-ink-soft"}`}>
                {preset.blurb}
              </span>
            </button>
          );
        })}
      </div>

      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,0.92fr)_minmax(0,1.08fr)]">
        <form onSubmit={onSubmit} className="flex flex-col gap-4 border border-rule bg-paper p-4 sm:p-5">
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="shape">Answer shape</Label>
              <select
                id="shape"
                value={shape}
                onChange={(event) => {
                  setShape(event.target.value as Shape);
                  setPresetId("custom");
                }}
                className="h-9 border border-rule bg-background px-2 text-sm"
              >
                {SHAPES.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
              </select>
            </div>
            {shape === "categorical" ? (
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="strategy">Choice head</Label>
                <select
                  id="strategy"
                  value={strategy}
                  onChange={(event) => setStrategy(event.target.value)}
                  className="h-9 border border-rule bg-background px-2 text-sm"
                >
                  <option value="auto">Auto — slice if the label is one token, else letters</option>
                  <option value="slice">Force a direct slice</option>
                  <option value="letter">Letter code, one position</option>
                  <option value="margin">Yes/no margin per option</option>
                  <option value="prototype">Prototype from the label&apos;s rows</option>
                </select>
              </div>
            ) : (
              <p className="self-end text-sm leading-snug text-ink-soft">{shapeHelp(shape)}</p>
            )}
          </div>

          <div className="flex flex-col gap-1.5">
            <Label htmlFor="context">State</Label>
            <Textarea
              id="context"
              value={context}
              onChange={(event) => setContext(event.target.value)}
              rows={8}
              className="min-h-40 resize-y border-rule bg-background text-sm leading-relaxed"
            />
          </div>

          {shape !== "schema" && shape !== "questions" ? (
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="question">Question</Label>
              <Textarea
                id="question"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                rows={2}
                className="border-rule bg-background text-sm"
              />
            </div>
          ) : null}

          {shape === "categorical" || shape === "multilabel" ? (
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="options">{shape === "multilabel" ? "Flags, one per line" : "Options, one per line"}</Label>
              <Textarea
                id="options"
                value={options}
                onChange={(event) => setOptions(event.target.value)}
                rows={4}
                className="border-rule bg-background font-mono text-sm"
              />
            </div>
          ) : null}

          {shape === "ordinal" ? (
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="levels">Levels, low to high, one per line</Label>
              <Textarea
                id="levels"
                value={levels}
                onChange={(event) => setLevels(event.target.value)}
                rows={5}
                className="border-rule bg-background font-mono text-sm"
              />
            </div>
          ) : null}

          {shape === "schema" ? (
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="schema">JSON schema</Label>
              <Textarea
                id="schema"
                value={schema}
                onChange={(event) => setSchema(event.target.value)}
                rows={16}
                className="border-rule bg-background font-mono text-[13px] leading-relaxed"
              />
            </div>
          ) : null}

          {shape === "questions" ? (
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="questions">Questions by id</Label>
              <Textarea
                id="questions"
                value={questions}
                onChange={(event) => setQuestions(event.target.value)}
                rows={18}
                className="border-rule bg-background font-mono text-[13px] leading-relaxed"
              />
            </div>
          ) : null}

          {examples?.length ? (
            <p className="border border-rule bg-warn-soft px-3 py-2 text-sm leading-relaxed text-foreground">
              {examples.length} labeled states go with this request. A bias-and-temperature correction and a ridge
              probe are fit in closed form and kept only if held-out loss and accuracy beat the frozen slice.
            </p>
          ) : null}

          <div className="flex items-center justify-between gap-3">
            <Button type="submit" disabled={!ready || running || !context.trim()} className="rounded-none bg-foreground text-paper hover:bg-foreground/90">
              {running ? "Reading…" : examples?.length ? "Fit and read" : "Compile and read"}
            </Button>
            <p className="text-right font-mono text-[11px] text-ink-soft">
              {ready ? "weights resident" : "waiting on weights"}
            </p>
          </div>
          {error ? <p className="border border-gen/40 bg-gen-soft px-3 py-2 text-sm text-gen">{error}</p> : null}
        </form>

        <section className="flex min-h-[28rem] flex-col gap-4 border border-rule bg-paper p-4 sm:p-5">
          {!result ? <Idle ready={ready} error={health?.error ?? null} /> : <Trace result={result} openPrompt={openPrompt} setOpenPrompt={setOpenPrompt} />}
        </section>
      </div>

      <section className="grid gap-px border border-rule bg-rule sm:grid-cols-2 lg:grid-cols-3">
        {NOTES.map((note) => (
          <article key={note.title} className="bg-background px-4 py-4">
            <h2 className="font-serif text-lg">{note.title}</h2>
            <p className="mt-2 text-sm leading-relaxed text-ink-soft">{note.body}</p>
          </article>
        ))}
      </section>
    </div>
  );
}

function Idle({ ready, error }: { ready: boolean; error: string | null }) {
  return (
    <div className="flex h-full flex-col justify-between gap-6">
      <div>
        <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink-soft">No readout yet</p>
        <p className="mt-3 font-serif text-2xl leading-snug">
          {error
            ? "The trunk did not load."
            : ready
              ? "Ask a closed question. The compiler will pick the head."
              : "Loading the frozen model. The first answer waits for the weights."}
        </p>
        {error ? <p className="mt-3 text-sm text-gen">{error}</p> : null}
      </div>
      <ol className="space-y-3 text-sm leading-relaxed text-ink-soft">
        <li>1. The state is prefilled once.</li>
        <li>2. Each field branches at its own answer boundary. Nothing is sampled.</li>
        <li>3. The head is a few rows of the unembedding, or a short decode if the field is an open string.</li>
      </ol>
    </div>
  );
}

function Trace({
  result,
  openPrompt,
  setOpenPrompt,
}: {
  result: DecideResponse;
  openPrompt: string | null;
  setOpenPrompt: (id: string | null) => void;
}) {
  const generated = result.fields.reduce((sum, field) => sum + field.generated_tokens, 0);
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end justify-between gap-3 border-b border-rule pb-3">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink-soft">Compiled graph</p>
          <p className="mt-1 text-sm text-foreground">
            Prefill {result.prefill_tokens ?? result.shared_prefix_tokens} tokens
            {result.fields.length > 1 ? ` (${result.shared_prefix_tokens} shared across ${result.fields.length} fields)` : ""} ·{" "}
            {result.elapsed_ms.toFixed(0)} ms · {generated === 0 ? "nothing generated" : `${generated} generated tokens`}
          </p>
        </div>
        <p className="font-mono text-[11px] text-ink-soft">
          d={result.hidden_size} · V={result.vocab_size.toLocaleString()}
        </p>
      </div>

      <div className="border border-rule px-3 py-2">
        <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-ink-soft">One prefill of the state</p>
        <ul className="mt-2 space-y-1">
          {result.fields.map((field) => (
            <li key={field.id} className="flex flex-wrap items-baseline justify-between gap-2 text-sm">
              <span className="font-mono text-[13px]">{field.id}</span>
              <span className={field.generated_tokens > 0 ? "text-gen" : "text-read"}>
                {headLabel(field.head)}
                {field.generated_tokens > 0
                  ? ` · ${field.generated_tokens} tokens`
                  : field.rows_scored
                    ? ` · ${field.rows_scored} rows`
                    : ""}
              </span>
            </li>
          ))}
        </ul>
      </div>

      {result.plan ? <PlanCard plan={result.plan} /> : null}

      {result.notes.map((note) => (
        <p key={note} className="text-sm leading-relaxed text-ink-soft">
          {note}
        </p>
      ))}

      {result.fit ? <FitCard fit={result.fit} /> : null}

      <div className="flex flex-col gap-3">
        {result.fields.map((field) => (
          <FieldCard
            key={field.id}
            field={field}
            vocab={result.vocab_size}
            open={openPrompt === field.id}
            onToggle={() => setOpenPrompt(openPrompt === field.id ? null : field.id)}
          />
        ))}
      </div>
    </div>
  );
}

function PlanCard({ plan }: { plan: Plan }) {
  return (
    <div className="border border-rule px-3 py-2">
      <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-ink-soft">Plan</p>
      <p className="mt-1 text-sm">{plan.summary}</p>
      <ul className="mt-2 space-y-1.5">
        {plan.fields.map((field) => (
          <li key={field.id} className="font-mono text-[12px] leading-relaxed">
            <span className={field.skipped ? "text-ink-soft line-through" : "text-foreground"}>{field.id}</span>
            <span className="text-ink-soft">
              {" "}
              · {field.type ?? field.head} · {field.branches} branch{field.branches === 1 ? "" : "es"}
              {field.combine ? ` · ${field.combine}` : ""}
              {field.decode ? ` · ${field.decode}` : ""}
              {field.depth ? ` · ${field.depth}` : ""}
              {field.source ? ` · ${field.source}` : ""}
              {field.depends_on ? ` · if ${field.depends_on.question} = ${JSON.stringify(field.depends_on.when)}` : ""}
              {field.continues ? ` · continues ${field.continues}` : field.include_answers?.length ? ` · sees ${field.include_answers.join(", ")}` : ""}
              {field.skipped ? " · skipped" : ""}
            </span>
          </li>
        ))}
      </ul>
      {plan.stages.length > 1 ? (
        <p className="mt-2 font-mono text-[11px] text-ink-soft">
          stages: {plan.stages.map((stage) => stage.join(", ")).join(" → ")}
        </p>
      ) : null}
    </div>
  );
}

function FieldCard({
  field,
  vocab,
  open,
  onToggle,
}: {
  field: FieldResult;
  vocab: number;
  open: boolean;
  onToggle: () => void;
}) {
  const decoding = field.generated_tokens > 0;
  return (
    <article className={`border px-3 py-3 ${decoding ? "border-gen/40 bg-gen-soft/40" : "border-rule"}`}>
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-ink-soft">{field.id}</p>
          <p className="mt-1 font-serif text-2xl leading-tight">{field.skipped ? "skipped" : formatAnswer(field.answer)}</p>
          {field.score !== null ? (
            <p className="mt-1 font-mono text-xs text-ink-soft">expected level {field.score.toFixed(2)}</p>
          ) : null}
        </div>
        <Badge variant="outline" className={decoding ? "rounded-none border-gen text-gen" : "rounded-none border-read text-read"}>
          {headLabel(field.head)}
        </Badge>
      </div>

      {field.probabilities ? (
        <ul className="mt-3 space-y-1.5">
          {Object.entries(field.probabilities).map(([label, probability]) => (
            <li key={label} className="grid grid-cols-[minmax(0,1fr)_3.5rem] items-center gap-2 text-sm">
              <div>
                <div className="mb-0.5 flex justify-between gap-3">
                  <span className="truncate">
                    {field.letters?.[label] ? <span className="mr-2 font-mono text-ink-soft">{field.letters[label]}</span> : null}
                    {label}
                  </span>
                </div>
                <div className="h-1.5 bg-rule">
                  <div className="h-1.5 bg-read" style={{ width: `${Math.max(0, Math.min(100, probability * 100))}%` }} />
                </div>
              </div>
              <span className="text-right font-mono text-xs">{probability.toFixed(2)}</span>
            </li>
          ))}
        </ul>
      ) : null}

      {field.judgments ? (
        <p className="mt-2 font-mono text-xs text-ink-soft">
          {Object.entries(field.judgments)
            .map(([side, probability]) => `${side} criterion fits ${probability.toFixed(2)}`)
            .join(" · ")}
          {field.ambiguous ? " · the two judgments conflict" : ""}
        </p>
      ) : null}

      <p className="mt-3 text-sm leading-relaxed text-ink-soft">{field.reason}</p>
      {field.warning ? <p className="mt-2 text-sm leading-relaxed text-warn">{field.warning}</p> : null}

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[11px] text-ink-soft">
        {field.rows_scored > 0 ? (
          <span>
            {field.rows_scored} rows scored, not {vocab.toLocaleString()}
          </span>
        ) : null}
        {field.sequences > 1 ? <span>{field.sequences} branches</span> : null}
        {field.allowed_mass !== null ? <span>allowed mass {(field.allowed_mass * 100).toFixed(1)}%</span> : null}
        {field.confidence !== null ? <span>confidence {field.confidence.toFixed(2)}</span> : null}
        <button type="button" onClick={onToggle} className="underline decoration-rule underline-offset-2">
          {open ? "hide boundary" : "answer boundary"}
        </button>
      </div>
      {open ? <pre className="mt-2 overflow-x-auto whitespace-pre-wrap border border-rule bg-background p-2 font-mono text-[12px] leading-relaxed">{field.prompt}</pre> : null}
    </article>
  );
}

function FitCard({ fit }: { fit: NonNullable<DecideResponse["fit"]> }) {
  if (fit.fields?.length) {
    return (
      <div className="flex flex-col gap-2">
        {fit.fields.map((record) => (
          <FitCard key={record.field ?? record.note} fit={record} />
        ))}
      </div>
    );
  }
  return (
    <div className="border border-rule bg-warn-soft/50 px-3 py-3">
      <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-ink-soft">
        Fit {fit.field ? `· ${fit.field}` : ""} · {fit.chosen.replaceAll("_", " ")}
      </p>
      <p className="mt-1 text-sm leading-relaxed">{fit.note}</p>
      <p className="mt-2 font-mono text-[11px] text-ink-soft">
        {fit.n_examples ? `${fit.n_examples} labels` : ""}
        {fit.zero_shot_loo_nll !== undefined ? ` · slice nll ${fit.zero_shot_loo_nll.toFixed(3)}` : ""}
        {fit.affine_loo_nll !== undefined ? ` · affine nll ${fit.affine_loo_nll.toFixed(3)}` : ""}
        {fit.ridge_loo_nll !== undefined && fit.ridge_loo_nll !== null ? ` · ridge nll ${fit.ridge_loo_nll.toFixed(3)}` : ""}
      </p>
      <p className="mt-1 font-mono text-[11px] text-ink-soft">
        {fit.zero_shot_loo_accuracy !== undefined ? `held-out accuracy: slice ${pct(fit.zero_shot_loo_accuracy)}` : ""}
        {fit.affine_loo_accuracy !== undefined ? ` · affine ${pct(fit.affine_loo_accuracy)}` : ""}
        {fit.ridge_loo_accuracy !== undefined && fit.ridge_loo_accuracy !== null ? ` · ridge ${pct(fit.ridge_loo_accuracy)}` : ""}
        {fit.temperature && fit.temperature !== 1 ? ` · T=${fit.temperature.toFixed(2)}` : ""}
        {fit.bias
          ? ` · bias ${Object.entries(fit.bias)
              .map(([label, value]) => `${label} ${value >= 0 ? "+" : ""}${value.toFixed(2)}`)
              .join(", ")}`
          : ""}
      </p>
    </div>
  );
}

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function lines(value: string): string[] {
  return value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function shapeHelp(shape: Shape): string {
  if (shape === "boolean") return "Sigmoid of the Yes − No margin. One position.";
  if (shape === "ordinal") return "Softmax over the digit rows. The score is the expected level.";
  if (shape === "multilabel") return "Each flag is its own yes/no. They are not forced to sum to one.";
  if (shape === "questions") return "A map of typed questions. depends_on skips a question unless its parent matches; include_answers shows it earlier answers.";
  if (shape === "open") return "No rows to read. The model answers as an ordinary chat turn and pays for every token.";
  return "Each property becomes its own head. Closed fields share the prefill.";
}

function describeHealth(health: Health | null): string {
  if (!health) return "API not reachable yet.";
  if (health.error) return health.error;
  if (!health.loaded) return `Loading ${health.model}…`;
  return `${health.model} · ${health.device} · d=${health.hidden_size}`;
}

const NOTES = [
  {
    title: "Yes / no",
    body: "Log-sum-exp of the Yes tokens minus the same for No, then a sigmoid. Glance reads a photograph this way. The rows already live in the unembedding.",
  },
  {
    title: "One of",
    body: "If every label is a single token, slice those rows. If not, write the options as letters and slice the letter rows, which is the Simple Jev and OpenJev choice head.",
  },
  {
    title: "A rating",
    body: "An argmax throws away the rest of the rubric. Softmax the digit rows and report the expected level. A 2.4 is a different answer from a confident 2.",
  },
  {
    title: "Flags",
    body: "A softmax would crown one flag and suppress the others. Each flag is its own yes/no margin. Several can be true, and the probabilities do not sum to one.",
  },
  {
    title: "A schema",
    body: "Boolean, choice, rating, flags, and a copied span are different heads on one shared prefill. Only the open string pays for generation.",
  },
  {
    title: "A fitted head",
    body: "With a few labels, a per-class bias plus temperature, or a ridge probe on the answer-position hidden state, can correct the slice. The bias is what moves a decision boundary. Either is kept only when held-out loss and accuracy say so. The trunk is not updated.",
  },
];

export type FieldResult = {
  id: string;
  head: string;
  kind: string;
  reason: string;
  answer: string | boolean | string[] | number | null;
  confidence: number | null;
  probabilities: Record<string, number> | null;
  score: number | null;
  letters: Record<string, string> | null;
  allowed_mass: number | null;
  rows_scored: number;
  sequences: number;
  generated_tokens: number;
  prompt: string;
  logits: Record<string, number> | null;
  weak_reading: boolean;
  warning: string | null;
};

export type FitRecord = {
  field?: string;
  chosen: string;
  temperature?: number;
  bias?: Record<string, number> | null;
  n_examples?: number;
  skipped?: number;
  zero_shot_loo_nll?: number;
  affine_loo_nll?: number;
  ridge_loo_nll?: number | null;
  zero_shot_loo_accuracy?: number;
  affine_loo_accuracy?: number;
  ridge_loo_accuracy?: number | null;
  note: string;
  fields?: FitRecord[];
};

export type DecideResponse = {
  model: string;
  elapsed_ms: number;
  hidden_size: number;
  vocab_size: number;
  shared_prefix_tokens: number;
  prefill_tokens?: number;
  generated_tokens?: number;
  fields: FieldResult[];
  fit: FitRecord | null;
  notes: string[];
};

export type Health = {
  loaded: boolean;
  loading: boolean;
  error: string | null;
  model: string;
  hidden_size: number | null;
  vocab_size: number | null;
  device: string | null;
};

export type Shape = "boolean" | "categorical" | "ordinal" | "multilabel" | "open" | "schema";

export type Example = {
  context: string;
  label?: string | boolean | number;
  answers?: Record<string, string | boolean | number | string[]>;
};

export type Preset = {
  id: string;
  name: string;
  blurb: string;
  shape: Shape;
  context: string;
  question: string;
  options: string;
  levels: string;
  schema: string;
  strategy: string;
  examples: Example[] | null;
};

export const HEAD_LABEL: Record<string, string> = {
  binary_margin: "Yes/No margin",
  vocab_slice: "Unembedding slice",
  letter_slice: "Letter slice",
  option_margin: "Per-option margin",
  ordinal_expectation: "Level expectation",
  multilabel_margin: "Independent flags",
  extractive_decode: "Quote from the state",
  short_decode: "Short generation",
  chat_decode: "Plain chat answer",
  ridge_probe: "Fitted ridge probe",
  prototype: "Label prototype",
};

export function headLabel(head: string): string {
  if (head.endsWith("+affine")) {
    const base = head.slice(0, -"+affine".length);
    return `${HEAD_LABEL[base] ?? base} + fitted bias`;
  }
  return HEAD_LABEL[head] ?? head;
}

export function formatAnswer(value: FieldResult["answer"]): string {
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value)) return value.length ? value.join(", ") : "none";
  if (value === null || value === undefined) return "—";
  return String(value);
}

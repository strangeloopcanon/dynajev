import type { Preset } from "./types";

const ticket = `Subject: refund and a broken mug

Hi — the refund for order 4412 was posted on Tuesday, thank you. The replacement mug still arrived smashed. The box was crushed and the pieces are in the photo. Please send another one this week. This is not a billing dispute.`;

export const SCHEMA = `{
  "type": "object",
  "properties": {
    "refund_sent": {
      "type": "boolean",
      "description": "Has the refund already been sent?"
    },
    "need": {
      "description": "What does the customer need next?",
      "enum": ["a replacement mug", "a tracking update", "a billing review"]
    },
    "urgency": {
      "type": "integer",
      "minimum": 0,
      "maximum": 4,
      "description": "How urgent is this, from 0 ignore to 4 page someone?"
    },
    "flags": {
      "type": "array",
      "description": "Which flags apply?",
      "items": { "enum": ["grateful", "damaged", "legal"] }
    },
    "quote": {
      "type": "string",
      "description": "Quote the short span that says the mug arrived smashed.",
      "x-readout": "extract"
    }
  }
}`;

export const QUESTIONS = `{
  "damaged": { "type": "noul", "instructions": "Did the item arrive damaged?" },
  "flags": {
    "type": "flags",
    "instructions": "Which of these describe the email?",
    "options": ["grateful", "damaged", "legal"]
  },
  "remedy": {
    "type": "choice",
    "instructions": "What should we send?",
    "options": ["a replacement", "a refund", "an apology only"],
    "depends_on": { "question": "damaged", "when": true },
    "include_answers": ["damaged"]
  },
  "legal_review": {
    "type": "noul",
    "instructions": "Does this need legal review?",
    "depends_on": { "question": "flags", "when": "legal" }
  },
  "resolved": {
    "type": "noul",
    "instructions": "Has the customer's problem been resolved?",
    "criteria": {
      "true": "The customer says the problem is fixed and nothing more is needed.",
      "false": "The customer still needs something done."
    }
  }
}`;

export const PRESETS: Preset[] = [
  {
    id: "schema",
    name: "Mixed schema",
    blurb: "Five heads, one prefill",
    shape: "schema",
    context: ticket,
    question: "",
    options: "",
    levels: "",
    schema: SCHEMA,
    strategy: "auto",
    examples: null,
  },
  {
    id: "questions",
    name: "Typed questions",
    blurb: "Dependencies, one plan",
    shape: "questions",
    context: ticket,
    question: "",
    options: "",
    levels: "",
    schema: SCHEMA,
    questions: QUESTIONS,
    strategy: "auto",
    examples: null,
  },
  {
    id: "boolean",
    name: "Yes or no",
    blurb: "Two rows of the unembedding",
    shape: "boolean",
    context: ticket,
    question: "Has the refund already been sent?",
    options: "",
    levels: "",
    schema: SCHEMA,
    strategy: "auto",
    examples: null,
  },
  {
    id: "slice",
    name: "One token",
    blurb: "The label is already a token",
    shape: "categorical",
    context: "The bicycle in the photo is red. The basket is empty.",
    question: "What color is the bicycle?",
    options: "red\nblue\ngreen",
    levels: "",
    schema: SCHEMA,
    strategy: "auto",
    examples: null,
  },
  {
    id: "letter",
    name: "A phrase",
    blurb: "Letters, because the label is long",
    shape: "categorical",
    context: ticket,
    question: "What does the customer need next?",
    options: "a replacement mug\na tracking update\na billing review",
    levels: "",
    schema: SCHEMA,
    strategy: "auto",
    examples: null,
  },
  {
    id: "rating",
    name: "A rating",
    blurb: "Expected level, not a winner",
    shape: "ordinal",
    context: "Checkout has been down for every customer since 8am and the status page is silent.",
    question: "How urgent is this?",
    options: "",
    levels: "ignore\nlater today\ntoday\nnow\npage someone",
    schema: SCHEMA,
    strategy: "auto",
    examples: null,
  },
  {
    id: "flags",
    name: "Flags",
    blurb: "Sigmoids that do not compete",
    shape: "multilabel",
    context: ticket,
    question: "Which of these describe the email?",
    options: "grateful\ndamaged\nlegal",
    levels: "",
    schema: SCHEMA,
    strategy: "auto",
    examples: null,
  },
  {
    id: "open",
    name: "Open question",
    blurb: "No answer set, so it just generates",
    shape: "open",
    context: ticket,
    question: "What should we do next?",
    options: "",
    levels: "",
    schema: SCHEMA,
    strategy: "auto",
    examples: null,
  },
  {
    id: "fit",
    name: "Fit a head",
    blurb: "Leave-one-out on four labels",
    shape: "categorical",
    context: "Appreciate the quick refund, that helped a lot.",
    question: "What is the tone of this note?",
    options: "grateful\nangry",
    levels: "",
    schema: SCHEMA,
    strategy: "auto",
    examples: [
      { context: "Thanks so much, the refund landed this morning.", label: "grateful" },
      { context: "Really appreciate you fixing this.", label: "grateful" },
      { context: "Thank you, that was fast.", label: "grateful" },
      { context: "This is unacceptable and I want a manager.", label: "angry" },
      { context: "Furious. Nothing has been resolved.", label: "angry" },
      { context: "Stop ignoring me. This is outrageous.", label: "angry" },
    ],
  },
];

"""Run a few closed questions on the default trunk and print the compiled heads."""

import json

from readhead.compile import DecideIn
from readhead.engine import Readhead
from readhead.trunk import Trunk

MODEL = "Qwen/Qwen3.5-2B"


def main() -> None:
    trunk = Trunk.load(MODEL)
    engine = Readhead(trunk)
    requests = [
        DecideIn(
            context="We sent the refund on Tuesday. It posted to the card ending 4412.",
            question="Has the refund already been sent?",
            type="boolean",
        ),
        DecideIn(
            context="The bicycle in the photo is red. The basket is empty.",
            question="What color is the bicycle?",
            options=["red", "blue", "green"],
        ),
        DecideIn(
            context="The mug arrived smashed. The box was crushed.",
            question="What does the customer need?",
            options=["a replacement mug", "a tracking update", "a billing review"],
        ),
        DecideIn.model_validate(
            {
                "context": "We sent the refund on Tuesday. The mug arrived smashed. Please send another this week.",
                "schema": {
                    "type": "object",
                    "properties": {
                        "refund_sent": {"type": "boolean", "description": "Has the refund already been sent?"},
                        "need": {
                            "description": "What does the customer need next?",
                            "enum": ["a replacement mug", "a tracking update"],
                        },
                        "urgency": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 4,
                            "description": "How urgent is this, from 0 ignore to 4 page someone?",
                        },
                        "quote": {
                            "type": "string",
                            "description": "Quote the short span that says the mug arrived smashed.",
                        },
                    },
                },
            }
        ),
    ]
    for req in requests:
        result = engine.decide(req)
        brief = {
            "ms": result["elapsed_ms"],
            "shared": result["shared_prefix_tokens"],
            "fields": [
                {
                    "id": field["id"],
                    "head": field["head"],
                    "answer": field["answer"],
                    "score": field["score"],
                    "probs": field["probabilities"],
                    "mass": field["allowed_mass"],
                    "rows": field["rows_scored"],
                    "generated": field["generated_tokens"],
                    "warning": field["warning"],
                }
                for field in result["fields"]
            ],
        }
        print(json.dumps(brief, indent=2))
        print("---")


if __name__ == "__main__":
    main()

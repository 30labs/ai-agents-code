"""Parts of chapter 8's context examples that it describes."""

from __future__ import annotations

from typing import Any

from loop_parts import Model, Step, Turn

CATALOG = {"service": "catalog"}


def _logs(query: str) -> Step:
    return (
        "search_logs",
        {**CATALOG, "query": query, "detail": "detailed"},
    )


def catalog_reads() -> list[Step]:
    """A long investigation of `long-catalog`, as a script.

    Nine reads, six of them detailed log searches that
    return up to fifty lines each.
    """
    return [
        ("recent_changes", CATALOG),
        _logs(""),
        _logs("request"),
        _logs("batch"),
        ("get_metrics", {**CATALOG, "detail": "detailed"}),
        _logs("completed"),
        _logs("catalog"),
        _logs("error serialization"),
        ("search_runbooks", {"query": "serialization deploy"}),
    ]


def watched(model: Model, sizes: list[int]) -> Model:
    "Record each request's input tokens in `sizes`."

    def ask(messages: list[dict[str, Any]]) -> Turn:
        turn = model(messages)
        sizes.append(turn.input_tokens)
        return turn

    return ask


COMPACT = (
    "You summarise an on-call investigation so it can"
    " continue in a fresh context. Reply with JSON only,"
    " with these keys: 'alert' (one line), 'suspect' (the"
    " likeliest cause so far, or ''), 'evidence' (short"
    " facts, each with the time or id it rests on),"
    " 'ruled_out' (hypotheses the evidence rejected),"
    " 'ids' (every deploy id, flag name and service named"
    " in the tool results), 'next' (the next read to make,"
    " or ''). Keep ids exactly as written. Leave out log"
    " lines that only repeat routine activity."
)


def transcript(messages: list[dict[str, Any]]) -> str:
    "The history as plain text, one entry per message."
    out = []
    for message in messages[1:]:
        role = message["role"]
        for call in message.get("tool_calls", []):
            fn = call["function"]
            out.append(f"call {fn['name']} {fn['arguments']}")
        if message.get("content"):
            out.append(f"{role}: {message['content']}")
    return "\n".join(out)


def compact_request(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    "Ask for the history as a compact state, in one request."
    return [
        {"role": "system", "content": COMPACT},
        {"role": "user", "content": transcript(messages)},
    ]


def notes_message(state_json: str) -> dict[str, Any]:
    "The compact state, as the message a new context opens on."
    return {
        "role": "user",
        "content": "Notes from earlier in this incident,"
        " written before older results were removed: "
        + state_json,
    }


"""Lenient JSON for model output: code fences, text around the object, and unbalanced brackets.

LLMs occasionally emit one stray or missing bracket (seen from gpt-oss on Groq: an extra "}" closing a list
item). repair_brackets drops closers that don't match the open bracket and appends the missing ones;
anything else still fails loudly.
"""

import json
import re

_CLOSE = {"}": "{", "]": "["}
_OPEN = {"{": "}", "[": "]"}


def repair_brackets(text: str) -> str:
    out, stack, in_str, escaped = [], [], False, False
    for ch in text:
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in _OPEN:
            stack.append(ch)
        elif ch in _CLOSE:
            if not stack or stack[-1] != _CLOSE[ch]:
                continue  # stray closer: drop it
            stack.pop()
        out.append(ch)
    return "".join(out) + "".join(_OPEN[b] for b in reversed(stack))


def loads_lenient(text: str):
    """Parse the first JSON object in text, repairing bracket mistakes if needed. Raises ValueError."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    end = text.rfind("}")
    candidate = text[start:end + 1] if end > start else text[start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as first:
        try:
            return json.loads(repair_brackets(text[start:]))
        except json.JSONDecodeError:
            raise ValueError(f"invalid JSON: {first}") from None

"""Malformed model JSON: bracket repair, Groq's json_validate_failed, and the agent planner surviving both."""

import httpx
import openai
import pytest

from app.jsonfix import loads_lenient, repair_brackets

# The plan gpt-oss returned on 2026-10-08 that Groq rejected (400 json_validate_failed): one extra "}"
# after the first sketch ("...false}]}}},{"title"...).
GROQ_FAILED = (
    '{"summary":"Search for risky interactions between people and forklifts, including approaching, head‑on, '
    'and missing spotter scenarios","sketches":[{"title":"Person Approaches Forklift","why":"A person walking toward '
    'a stationary forklift at the far end can collide","sketch":{"objects":[{"id":"fork1","label":"forklift",'
    '"start_box":{"x":0.48,"y":0.08,"w":0.04,"h":0.10},"end_box":{"x":0.48,"y":0.08,"w":0.04,"h":0.10},"path":null,'
    '"absent":false},{"id":"p1","label":"person","start_box":{"x":0.48,"y":0.75,"w":0.05,"h":0.22},"end_box":{"x":0.48,'
    '"y":0.60,"w":0.05,"h":0.22},"path":[[0.505,0.86],[0.505,0.71]],"absent":false}]}}},{"title":"Head‑On Forklift '
    'and Person","why":"Both forklift and person move toward each other, creating a head‑on collision risk",'
    '"sketch":{"objects":[{"id":"fork1","label":"forklift","start_box":{"x":0.48,"y":0.08,"w":0.04,"h":0.10},'
    '"end_box":{"x":0.48,"y":0.20,"w":0.04,"h":0.10},"path":[[0.505,0.13],[0.505,0.25]],"absent":false},{"id":"p1",'
    '"label":"person","start_box":{"x":0.48,"y":0.75,"w":0.05,"h":0.22},"end_box":{"x":0.48,"y":0.50,"w":0.05,"h":0.22},'
    '"path":[[0.505,0.86],[0.505,0.61]],"absent":false}]}}},{"title":"Person Near Forklift Without Spotter","why":"A '
    'person operates close to a forklift while no spotter is present, a common safety violation","sketch":{"objects":'
    '[{"id":"fork1","label":"forklift","start_box":{"x":0.48,"y":0.08,"w":0.04,"h":0.10},"end_box":{"x":0.48,"y":0.08,'
    '"w":0.04,"h":0.10},"path":null,"absent":false},{"id":"p1","label":"person","start_box":{"x":0.60,"y":0.20,"w":0.05,'
    '"h":0.22},"end_box":{"x":0.60,"y":0.30,"w":0.05,"h":0.22},"path":[[0.625,0.31],[0.625,0.41]],"absent":false},'
    '{"id":"spot1","label":"person","start_box":{"x":0.30,"y":0.40,"w":0.20,"h":0.15},"end_box":null,"path":null,'
    '"absent":true}]}}}]}'
)


def test_repairs_the_groq_plan():
    data = loads_lenient(GROQ_FAILED)
    assert [s["title"] for s in data["sketches"]] == [
        "Person Approaches Forklift", "Head‑On Forklift and Person", "Person Near Forklift Without Spotter"]


@pytest.mark.parametrize("text, expected", [
    ('{"a": [1, 2}', {"a": [1, 2]}),  # wrong closer
    ('{"a": {"b": 1}', {"a": {"b": 1}}),  # missing closer
    ('```json\n{"a": "x } ] y"}\n```', {"a": "x } ] y"}),  # brackets inside strings are left alone
    ('Here you go: {"a": 1} hope it helps', {"a": 1}),
])
def test_loads_lenient(text, expected):
    assert loads_lenient(text) == expected


def test_unrepairable_json_still_fails():
    with pytest.raises(ValueError):
        loads_lenient('{"a": 1,, "b": }')


def test_repair_is_a_no_op_on_valid_json():
    good = '{"a": [1, {"b": "c"}]}'
    assert repair_brackets(good) == good


def _groq_400(body: dict) -> openai.BadRequestError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return openai.BadRequestError("Error code: 400", response=httpx.Response(400, request=request), body=body)


def test_failed_generation_is_returned_by_chat_json(monkeypatch):
    import app.llm as llm

    err = _groq_400({"error": {"message": "Failed to generate JSON.", "type": "invalid_request_error",
                               "code": "json_validate_failed", "failed_generation": GROQ_FAILED}})
    assert llm.failed_generation(err) == GROQ_FAILED
    assert llm.failed_generation(_groq_400({"error": {"code": "other", "message": "response_format"}})) is None

    class Completions:
        def create(self, **kw):
            raise err

    class Client:
        chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr(llm, "get_client", lambda provider: (Client(), "m"))
    monkeypatch.setattr(llm, "provider_config", lambda provider: {"model": "m", "json_mode": True, "extra": {}})
    assert llm.chat_json([{"role": "user", "content": "plan"}], provider="groq") == GROQ_FAILED


def test_planner_survives_the_rejected_plan(monkeypatch):
    import app.agent.runner as runner

    monkeypatch.setattr(runner, "chat_json", lambda messages: GROQ_FAILED)
    plan = runner.plan_sketches("Find risky moments between people and the forklift.")
    assert len(plan["sketches"]) == 3
    third = plan["sketches"][2]["sketch"]
    assert [(o.label, o.absent) for o in third.objects] == [("forklift", False), ("person", False), ("person", True)]

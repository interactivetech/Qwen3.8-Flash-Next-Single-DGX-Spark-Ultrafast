"""Regression test for 0001-tool-args-anyof.patch, run in the image build.

Before the patch Hibana cast a parameter only by `schema["type"]`, so an Optional[list[str]] (anyOf [array, null])
came back as the JSON string '["a@b.c"]' and the tool rejected it. Seen in AgentDojo's create_calendar_event, where the
model then retried until it ran out of tokens.
"""
import json

from hibana.api.parsers.tools import parse_tool_calls

TOOLS = [{"type": "function", "function": {"name": "create_calendar_event", "parameters": {"type": "object", "properties": {
    "title": {"type": "string"},
    "participants": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
    "location": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    "count": {"type": ["integer", "null"]},
    "meta": {"oneOf": [{"type": "object"}, {"type": "null"}]},
    "flag": {"type": "boolean"},
    "tags": {"type": "array"},
}}}}]


def call(params: str) -> dict:
    text = f"<tool_call>\n<function=create_calendar_event>\n{params}</function>\n</tool_call>"
    _, calls = parse_tool_calls(text, TOOLS)
    return json.loads(calls[0]["function"]["arguments"])


def p(name: str, value: str) -> str:
    return f"<parameter={name}>\n{value}\n</parameter>\n"


args = call(p("participants", '["sarah.connor@gmail.com"]') + p("title", "Lunch") + p("count", "3")
            + p("meta", '{"a": 1}') + p("flag", "true") + p("tags", '["x"]') + p("location", "Room 1"))
assert args == {"participants": ["sarah.connor@gmail.com"], "title": "Lunch", "count": 3, "meta": {"a": 1},
                "flag": True, "tags": ["x"], "location": "Room 1"}, args
assert call(p("participants", "null"))["participants"] is None
assert call(p("count", "null"))["count"] is None
# A value that does not fit stays the model's text, so the tool's own error tells it what went wrong.
assert call(p("participants", "sarah.connor@gmail.com"))["participants"] == "sarah.connor@gmail.com"
assert call(p("location", '["a"]'))["location"] == '["a"]'
assert call(p("title", "null"))["title"] == "null"
print("0001-tool-args-anyof: ok")

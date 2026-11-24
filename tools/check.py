#!/usr/bin/env python3
"""Does Qwen3.8-Flash-Next on Hibana behave? Checks the model listing, that any model id answers, thinking off and on
(reasoning kept out of the answer), a tool call and its round trip, streaming, Dutch and image input (a generated
two-colour PNG, sent as a data URL) against a running server, and shows the serving profile and text and image speed,
optionally plus needle-in-a-haystack retrieval at long context.

  tools/check.py                         # functional checks
  tools/check.py --needle 32000 131072   # also hide a key in N tokens of filler and ask for it back
  tools/check.py --url http://<spark-ip>:8888

Exit code 1 if anything fails. Standard library only.
"""
import argparse
import base64
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import zlib
import struct


class Server:
    def __init__(self, url: str, model: str):
        self.base, self.model = url.rstrip("/"), model
        self.api_key = os.environ.get("HIBANA_API_KEY", "")

    def _open(self, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base + path, data, headers)
        try:
            return urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as err:
            sys.exit(f"{self.base}{path}: HTTP {err.code} {err.read().decode(errors='replace')[:400]}")
        except urllib.error.URLError as err:
            sys.exit(f"{self.base}{path}: {err.reason} (is the server up? docker compose ps)")

    def models(self) -> list[str]:
        payload = json.load(self._open("/v1/models"))
        return [item.get("id", "") for item in payload.get("data", [])]

    def ask(self, prompt: str, **body) -> dict:
        """One non-streamed turn; returns the message plus usage."""
        reply = json.load(self._open("/v1/chat/completions",
                                      {"model": self.model, "messages": [{"role": "user", "content": prompt}], **body}))
        return {**reply["choices"][0]["message"], "usage": reply.get("usage", {})}

    def stream(self, prompt: str, **body) -> tuple[str, str]:
        """One streamed turn; returns (reasoning, content) as they arrived in the deltas."""
        parts = {"reasoning": [], "content": []}
        body = {"model": self.model, "stream": True, "messages": [{"role": "user", "content": prompt}], **body}
        for raw in self._open("/v1/chat/completions", body):
            event = raw.decode().removeprefix("data:").strip()
            if not event or event == "[DONE]":
                continue
            for choice in json.loads(event).get("choices", []):
                delta = choice.get("delta", {})
                parts["reasoning"].append(delta.get("reasoning") or delta.get("reasoning_content") or "")
                parts["content"].append(delta.get("content") or "")
        return "".join(parts["reasoning"]), "".join(parts["content"])


def reasoning(msg: dict) -> str:
    return (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()


def clean(text: str | None) -> str:
    return (text or "").strip()


def model_listed(srv: Server):
    """Hibana lists the checkpoint it serves under its own id and takes any id in a request."""
    names = srv.models()
    return any("Flash-Next" in n for n in names), f"ids {names}"


def alias(srv: Server):
    msg = Server(srv.base, "local-ai").ask("Reply with the word ok.", max_tokens=256, temperature=0, **OFF)
    answer = clean(msg["content"])
    return "ok" in answer.lower(), f"model local-ai, answer {answer[:40]!r}"


OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def thinking_off(srv: Server):
    msg = srv.ask("What is 17 + 25? Digits only.", max_tokens=4096, temperature=0, **OFF)
    answer = clean(msg["content"])
    ok = "42" in answer and "<think>" not in answer and not reasoning(msg)
    return ok, f"answer {answer!r}, reasoning {len(reasoning(msg))} chars"


def thinking_on(srv: Server):
    """Thinking is on by default: the reasoning must land in reasoning_content, not in the answer."""
    msg = srv.ask("A notebook costs 4 euros and a pen costs 1.50 euros. "
                  "How much do 3 notebooks and 2 pens cost, in euros? Reply with the number only.",
                  max_tokens=16384, temperature=0.6)
    answer = clean(msg["content"])
    ok = "15" in answer and "<think>" not in answer and "</think>" not in answer and bool(reasoning(msg))
    return ok, f"answer {answer!r}, reasoning {len(reasoning(msg))} chars"


EVENT_TOOL = {"type": "function", "function": {
    "name": "create_event",
    "description": "Create a calendar event.",
    "parameters": {"type": "object", "required": ["title", "date"], "properties": {
        "title": {"type": "string"},
        "date": {"type": "string", "description": "ISO date, YYYY-MM-DD"}}}}}
EVENT_ASK = "Add a calendar event titled Sprint Review on 12 March 2027."


def tool_call(srv: Server):
    msg = srv.ask(EVENT_ASK, tools=[EVENT_TOOL], tool_choice="auto", max_tokens=4096, temperature=0, **OFF)
    calls = msg.get("tool_calls") or []
    if not calls:
        return False, f"no tool call; content {clean(msg.get('content'))[:160]!r}"
    fn = calls[0].get("function", {})
    raw = fn.get("arguments", {})
    args = json.loads(raw) if isinstance(raw, str) else raw
    title = str(args.get("title", ""))
    ok = fn.get("name") == "create_event" and args.get("date") == "2027-03-12" and "Sprint Review" in title
    return ok, f"{fn.get('name')}({json.dumps(args, ensure_ascii=False)})"


def tool_round_trip(srv: Server):
    """The tool result goes back as a tool message; the model must answer from it, not call again."""
    call = {"id": "call_1", "type": "function",
            "function": {"name": "create_event", "arguments": json.dumps({"title": "Sprint Review", "date": "2027-03-12"})}}
    messages = [{"role": "user", "content": EVENT_ASK},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"ok": True, "event_id": "EV-4471"})}]
    reply = json.load(srv._open("/v1/chat/completions", {"model": srv.model, "messages": messages, "tools": [EVENT_TOOL],
                                                         "max_tokens": 4096, "temperature": 0, **OFF}))
    msg = reply["choices"][0]["message"]
    answer = clean(msg.get("content"))
    return "EV-4471" in answer and not msg.get("tool_calls"), f"answer {answer[:90]!r}"


def streaming(srv: Server):
    think, answer = srv.stream("What is the capital of Belgium? One word.", max_tokens=4096, temperature=0, **OFF)
    text = clean(answer)
    ok = "Brussels" in text and "<think>" not in text + think
    return ok, f"reasoning {len(think)} chars, answer {text[:70]!r}"


def dutch(srv: Server):
    """Answers in Dutch. The fact itself is model knowledge, so it is shown, not checked."""
    msg = srv.ask("Wat is de hoofdstad van de provincie West-Vlaanderen? Antwoord in één Nederlandse zin.",
                  max_tokens=4096, temperature=0, **OFF)
    answer = clean(msg["content"])
    return "hoofdstad" in answer.lower(), f"answer {answer[:90]!r}"


def serving(srv: Server):
    """The profile and speculation the server reports (/get_server_info). Shown, not checked."""
    info = json.load(srv._open("/get_server_info"))
    cfg = info.get("config", info)
    keys = ("profile", "proposer", "spec", "max_batch", "max_context", "accept_soft", "w3_decode", "vision",
            "draft_vocab")
    shown = {k: cfg.get(k) for k in keys if k in cfg} or {k: info[k] for k in list(info)[:6]}
    return None, json.dumps(shown, ensure_ascii=False)[:200]


def speed(srv: Server):
    """Output tokens per second for one request, prefill included. Shown, not checked."""
    start = time.time()
    msg = srv.ask("Write a Python function that parses ISO 8601 durations like P3DT4H5M into seconds, "
                  "with a docstring and three doctests.", max_tokens=1024, temperature=0, **OFF)
    n, took = msg["usage"].get("completion_tokens", 0), time.time() - start
    return None if not n else True, f"{n} tokens in {took:.1f} s = {n / took:.1f} tok/s"


def png(width: int, height: int, left: tuple, right: tuple) -> bytes:
    """A width x height RGB PNG, left half one colour and right half another (standard library only)."""
    row = b"\x00" + bytes(left) * (width // 2) + bytes(right) * (width - width // 2)
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(row * height)) + chunk(b"IEND", b"")


def image_message(text: str) -> list[dict]:
    url = "data:image/png;base64," + base64.b64encode(png(512, 512, (220, 30, 30), (30, 60, 220))).decode()
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}]}]


def image_input(srv: Server):
    """Image input (this recipe's vision patches): the left half is red, the right half blue."""
    reply = json.load(srv._open("/v1/chat/completions", {
        "model": srv.model, "max_tokens": 4096, "temperature": 0, **OFF,
        "messages": image_message("This image has two coloured halves. Name the colour of the left half, then the "
                                  "right half, as: left: <colour>, right: <colour>.")}))
    answer = clean(reply["choices"][0]["message"].get("content")).lower()
    ok = "red" in answer and "blue" in answer and answer.find("red") < answer.find("blue")
    return ok, f"answer {answer[:80]!r}"


def image_speed(srv: Server):
    """Output tokens per second for one image request, image encode and prefill included. Shown, not checked."""
    start = time.time()
    reply = json.load(srv._open("/v1/chat/completions", {
        "model": srv.model, "max_tokens": 1024, "temperature": 0, **OFF,
        "messages": image_message("Describe this image, then write a short poem about the two colours.")}))
    n, took = reply.get("usage", {}).get("completion_tokens", 0), time.time() - start
    return None if not n else True, f"{n} tokens in {took:.1f} s = {n / took:.1f} tok/s"


def needle(srv: Server, tokens: int):
    """Hide a key among ~`tokens` tokens of synthetic service logs, at a random depth, and ask for it."""
    rng = random.Random(tokens)
    key = f"{rng.randrange(10**7, 10**8)}"
    lines = [f"2026-10-08T{rng.randrange(24):02}:{rng.randrange(60):02}:{rng.randrange(60):02}Z "
             f"worker-{rng.randrange(64):02} {rng.choice(('GET', 'POST', 'PUT'))} /api/v2/"
             f"{rng.choice(('orders', 'users', 'stock', 'invoices'))}/{rng.randrange(10**6)} -> "
             f"{rng.choice((200, 200, 200, 201, 404, 503))} in {rng.randrange(2, 900)} ms"
             for _ in range(tokens // 40)]
    depth = rng.uniform(0.1, 0.9)
    lines.insert(int(len(lines) * depth), f"NOTICE maintenance key for tonight is {key}")
    start = time.time()
    msg = srv.ask("\n".join(lines) + "\n\nWhat is tonight's maintenance key? Reply with the key only.",
                  max_tokens=4096, temperature=0, **OFF)
    got, n = clean(msg["content"]), msg["usage"].get("prompt_tokens", 0)
    return key in got, f"{n:,} tokens, depth {depth:.0%}, {time.time() - start:.0f} s: expected {key}, got {got!r}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("HIBANA_URL", "http://127.0.0.1:8888"))
    ap.add_argument("--model", default=os.environ.get("HIBANA_MODEL", "Qwen3.8-Flash-Next"))
    ap.add_argument("--needle", type=int, nargs="*", default=[], metavar="TOKENS")
    args = ap.parse_args()

    srv = Server(args.url, args.model)
    checks = [(fn.__name__.replace("_", " "), fn) for fn in
              (model_listed, alias, thinking_off, thinking_on, tool_call, tool_round_trip, streaming, dutch, image_input, serving, speed,
               image_speed)]
    checks += [(f"needle {n:,}", lambda s, n=n: needle(s, n)) for n in args.needle]
    failures = 0
    for name, check in checks:
        ok, detail = check(srv)
        failures += ok is False
        print(f"{'skip' if ok is None else 'ok  ' if ok else 'FAIL'}  {name:20} {detail}", flush=True)
    sys.exit(failures > 0)


if __name__ == "__main__":
    main()

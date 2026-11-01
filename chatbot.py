import argparse
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient

BASE = Path(__file__).resolve().parent
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
HEADERS = {}

SYSTEM_PROMPT = """You are an assistant that explores a collection of 7,349 pet images \
(37 cat and dog breeds, Oxford-IIIT Pets) using a fine-tuned ResNet-18.
You MUST use the tools to get facts. Never invent numbers, breeds or image names.

Tools:
- classify(image): top-3 predicted breeds of one image.
- find_similar(image, k): the k most similar images (embedding nearest neighbours).
- filter_images(label, min_confidence, max_results): images predicted as a breed, with a minimum confidence (0-1, so 80% = 0.8). Also gives the total count.
- count_by(label): number of images predicted per breed (empty label = all breeds).
- most_confused(label): breeds the model confuses with a given breed (test set).

Rules:
- An image is an id ("42"), a file name ("Abyssinian_1") or a file path.
- Percentages must be converted to a 0-1 number.
- Several tools can be chained: use the output of one tool as input of the next.
- If a tool returns an error, tell the user and suggest a fix (e.g. valid breed names).
- Answer in the language of the user, briefly, using the tool results. When asked "why", \
give a short hypothesis (similar coat, colour, morphology) and say it is a hypothesis."""


def to_ollama_tools(mcp_tools):
    return [{"type": "function",
             "function": {"name": t.name, "description": t.description or "",
                          "parameters": t.inputSchema}} for t in mcp_tools]


@asynccontextmanager
async def open_session(host=None):
    """Lance mcp_server.py en sous-processus et renvoie (session MCP, client Ollama, tools)."""
    params = StdioServerParameters(command=sys.executable, args=[str(BASE / "mcp_server.py")],
                                   cwd=str(BASE))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = to_ollama_tools((await session.list_tools()).tools)
            client = AsyncClient(host=host or os.environ.get("OLLAMA_HOST"), headers=HEADERS or None)
            yield session, client, tools


async def ask(session, client, model, tools, messages, max_steps=6, verbose=True):
    """Boucle d'appel d'outils. Renvoie (réponse texte, liste des appels {tool, args, result})."""
    calls = []
    for _ in range(max_steps):
        resp = await client.chat(model=model, messages=messages, tools=tools,
                                 options={"temperature": 0})
        msg = resp.message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content or "", calls
        for tc in msg.tool_calls:
            name, args = tc.function.name, dict(tc.function.arguments)
            try:
                res = await session.call_tool(name, args)
                text = "\n".join(c.text for c in res.content if hasattr(c, "text"))
            except Exception as e:  # argument invalide, outil inconnu...
                text = json.dumps({"error": str(e)})
            if verbose:
                print(f"  [tool] {name}({json.dumps(args, ensure_ascii=False)})")
            calls.append({"tool": name, "args": args, "result": text})
            messages.append({"role": "tool", "tool_name": name, "content": text})
    return "(trop d'étapes sans réponse finale)", calls


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--host", default=None, help="URL Ollama (défaut : local / OLLAMA_HOST)")
    args = p.parse_args()

    async with open_session(args.host) as (session, client, tools):
        print(f"Modèle : {args.model} | outils : {[t['function']['name'] for t in tools]}")
        print("Posez vos questions (Ctrl+C ou 'quit' pour sortir).\n")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        while True:
            try:
                q = await asyncio.to_thread(input, "vous > ")
            except (EOFError, KeyboardInterrupt):
                break
            if q.strip().lower() in {"quit", "exit", "q"}:
                break
            if not q.strip():
                continue
            messages.append({"role": "user", "content": q})
            answer, _ = await ask(session, client, args.model, tools, messages)
            print(f"\nbot > {answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
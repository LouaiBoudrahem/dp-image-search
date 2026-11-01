"""Évaluation du chatbot : taux de bons appels d'outils sur chatbot_tests.json.

Pour chaque requête, on compare les appels d'outils du LLM aux appels attendus :
- outils corrects : l'ensemble des outils appelés est exactement l'ensemble attendu ;
- arguments corrects : chaque appel attendu a été fait avec les bons arguments
  (noms de races comparés avec la même recherche floue que le serveur MCP).

Usage :
    python eval_chatbot.py                       # modèle local par défaut (qwen2.5:7b)
    python eval_chatbot.py --model llama3.1:8b --host http://HOTE_CRIL:11434
Résultats : outputs/chatbot_eval_<modele>.csv et un résumé à l'écran.
"""
import argparse
import asyncio
import csv
import json
from pathlib import Path

from chatbot import DEFAULT_MODEL, SYSTEM_PROMPT, ask, open_session
from mcp_server import find_class

BASE = Path(__file__).resolve().parent


def same_image(a, b):
    stem = lambda s: Path(str(s).strip().replace("\\", "/")).stem.lower()
    return stem(a) == stem(b)


def same_label(a, b):
    if not str(a).strip() and not str(b).strip():
        return True
    return find_class(str(a)) is not None and find_class(str(a)) == find_class(str(b))


def same_conf(a, b):
    a = float(a)
    if a > 1:  # le serveur accepte aussi 90 pour 90 %
        a /= 100
    return abs(a - float(b)) < 1e-6


def args_match(expected, got):
    for key, want in expected.items():
        if key not in got:
            return False
        have = got[key]
        try:
            if key == "image" and not same_image(want, have):
                return False
            if key == "label" and not same_label(want, have):
                return False
            if key == "min_confidence" and not same_conf(have, want):
                return False
            if key == "k" and int(have) != int(want):
                return False
        except (TypeError, ValueError):
            return False
    if not expected and str(got.get("label", "")).strip():  # count_by() sans race
        return False
    return True


def score(expected, calls):
    called = [c["tool"] for c in calls]
    tools_ok = set(called) == {e["tool"] for e in expected}
    args_ok = tools_ok and all(
        any(c["tool"] == e["tool"] and args_match(e["args"], c["args"]) for c in calls)
        for e in expected)
    return tools_ok, args_ok


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--host", default=None, help="URL Ollama (défaut : local / OLLAMA_HOST)")
    p.add_argument("--tests", default=str(BASE / "chatbot_tests.json"))
    args = p.parse_args()

    tests = json.load(open(args.tests, encoding="utf-8"))
    rows = []
    async with open_session(args.host) as (session, client, tools):
        for i, t in enumerate(tests, 1):
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": t["query"]}]
            try:
                answer, calls = await ask(session, client, args.model, tools, messages, verbose=False)
            except Exception as e:  # erreur réseau / modèle
                answer, calls = f"ERREUR : {e}", []
            tools_ok, args_ok = score(t["expected"], calls)
            chain = len(t["expected"]) > 1
            rows.append({"id": i, "query": t["query"], "chain": chain,
                         "expected": json.dumps(t["expected"], ensure_ascii=False),
                         "called": json.dumps([{"tool": c["tool"], "args": c["args"]} for c in calls],
                                              ensure_ascii=False),
                         "tools_ok": tools_ok, "args_ok": args_ok, "answer": answer.strip()})
            mark = "OK " if args_ok else ("~  " if tools_ok else "KO ")
            print(f"{mark} [{i:2d}] {t['query'][:70]}")
            print(f"       appels : {[(c['tool'], c['args']) for c in calls]}")

    n = len(rows)
    single = [r for r in rows if not r["chain"]]
    chains = [r for r in rows if r["chain"]]
    pct = lambda rs, k: 100 * sum(r[k] for r in rs) / max(1, len(rs))
    summary = [
        f"Modèle : {args.model} | {n} requêtes ({len(single)} à 1 outil, {len(chains)} enchaînées)",
        f"Bons outils             : {pct(rows, 'tools_ok'):.0f} %",
        f"Bons outils + arguments : {pct(rows, 'args_ok'):.0f} %",
        f"  requêtes à 1 outil    : {pct(single, 'args_ok'):.0f} %",
        f"  requêtes enchaînées   : {pct(chains, 'args_ok'):.0f} %",
    ]
    print("\n" + "\n".join(summary))

    out = BASE / "outputs"
    out.mkdir(exist_ok=True)
    tag = args.model.replace(":", "_").replace("/", "_")
    with open(out / f"chatbot_eval_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / f"chatbot_eval_{tag}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
    print(f"\nDétails : outputs/chatbot_eval_{tag}.csv")


if __name__ == "__main__":
    asyncio.run(main())

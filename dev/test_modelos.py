"""Descobre QUAIS modelos a sua conta realmente pode invocar.

Uso: .venv\\Scripts\\python.exe dev\\test_modelos.py
Faz uma chamada mínima por modelo e imprime o que responde.
"""

import sys

sys.path.insert(0, ".")

from openai import OpenAI

from src.llm import get_client, get_modelo

# Modelos claramente não-texto (embedding, visão, segurança, código, imagem)
EXCLUIR = (
    "embed",
    "vision",
    "deplot",
    "neva",
    "vila",
    "nvclip",
    "kosmos",
    "fuyu",
    "guard",
    "safety",
    "content-safety",
    "topic-control",
    "reward",
    "code",
    "video-detector",
    "parse",
    "reward",
    "diffusion",
    "recurrentgemma",
    "starcoder",
    "riva-translate",
)

PING = "Responda apenas com a palavra: ok"

# Ordem de preferência: os melhores para gerar questões em PT-BR primeiro.
PRIORITARIOS = [
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/llama-3.1-nemotron-51b-instruct",
    "mistralai/mistral-large-2-instruct",
    "mistralai/mistral-large",
    "openai/gpt-oss-20b",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "z-ai/glm-5.3",
    "z-ai/glm-5.3-flash",
    "moonshotai/kimi-k2.6",
    "deepseek-ai/deepseek-v4.1-flash",
    "stepfun-ai/step-5-preview",
]


def candidatos(client: OpenAI) -> list[str]:
    ids = [m.id for m in client.models.list()]
    texto = [i for i in ids if not any(e in i for e in EXCLUIR)]
    # prioriza os que gostaríamos de usar, depois o resto
    pri = [m for m in PRIORITARIOS if m in texto]
    resto = [m for m in texto if m not in pri]
    return pri + resto


def testar(client: OpenAI, model_id: str) -> tuple[bool, str]:
    try:
        r = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": PING}],
            max_tokens=5,
            temperature=0,
        )
        return True, (r.choices[0].message.content or "").strip()
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"


def main() -> int:
    client = get_client()
    modelo_atual = get_modelo()
    print(f"Modelo no .env: {modelo_atual}\n")

    lista = candidatos(client)
    so_top = "--todos" not in sys.argv
    if so_top:
        lista = lista[: len(PRIORITARIOS)]
        print("Testando só os prioritários (use --todos para testar tudo).")
    print(f"Testando {len(lista)} modelos (pode demorar)...\n")

    ok: list[tuple[str, str]] = []
    falhas: list[tuple[str, str]] = []

    for i, m in enumerate(lista, 1):
        funciona, info = testar(client, m)
        marca = "OK " if funciona else "x  "
        print(f"  [{i:2}/{len(lista)}] {marca} {m}  -> {info}")
        (ok if funciona else falhas).append((m, info))

    print("\n" + "=" * 60)
    print(f"\nFUNCIONAM ({len(ok)}):")
    for m, _ in ok:
        print(f"  {m}")
    print(f"\nFALHARAM ({len(falhas)}):")
    for m, info in falhas:
        print(f"  {m}  ({info})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

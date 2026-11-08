"""Pinga os provedores gratuitos configurados no .env.

Uso:
  .venv\\Scripts\\python.exe dev\\test_providers.py
  .venv\\Scripts\\python.exe dev\\test_providers.py --provedor openrouter

Testa apenas provedores com chave configurada. Se o free tier responder 429,
para aquele provedor e mostra o erro (não há fallback pago).
"""

import os
import sys

sys.path.insert(0, ".")

from src.llm import chat, get_client, get_modelo, ollama_disponivel
from src.providers import PROVEDORES, ids_modelos, lista_provedores

PING = "Responda apenas com a palavra: ok"


def tem_chave(provedor: str) -> bool:
    p = PROVEDORES[provedor]
    if p.id == "ollama":
        return True
    chave = os.getenv(p.env_key, "").strip()
    return bool(chave) and not chave.endswith("...")


def modelos_do_teste(provedor: str) -> list[str]:
    if provedor in ("nvidia", "ollama"):
        try:
            return [get_modelo(provedor)]
        except RuntimeError:
            return ids_modelos(provedor)[:1]
    return ids_modelos(provedor)


def eh_429(erro: Exception) -> bool:
    status = getattr(erro, "status_code", None)
    return status == 429 or "429" in str(erro) or "rate limit" in str(erro).lower()


def testar(provedor: str, modelo: str) -> tuple[bool, str, bool]:
    try:
        saida = chat(
            [{"role": "user", "content": PING}],
            temperature=0,
            modelo=modelo,
            provedor=provedor,
        )
        return True, saida.strip()[:60], False
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}", eh_429(e)


def main() -> int:
    alvo = None
    if "--provedor" in sys.argv:
        i = sys.argv.index("--provedor")
        if i + 1 < len(sys.argv):
            alvo = sys.argv[i + 1].strip().lower()

    provedores = [alvo] if alvo else lista_provedores()
    if alvo and alvo not in PROVEDORES:
        print(f"[ERRO] provedor desconhecido: {alvo}")
        return 1

    testados = 0
    pulados = 0
    for provedor in provedores:
        p = PROVEDORES[provedor]
        print(f"\n=== {p.nome} ({provedor}) ===")
        if not tem_chave(provedor):
            print(f"  [PULADO] defina {p.env_key} no .env")
            pulados += 1
            continue
        if provedor == "ollama" and not ollama_disponivel():
            print("  [PULADO] Ollama não instalado/executando")
            pulados += 1
            continue
        try:
            get_client(provedor)
        except Exception as e:
            print(f"  [ERRO] {e}")
            continue

        for modelo in modelos_do_teste(provedor):
            testados += 1
            ok, info, limite = testar(provedor, modelo)
            marca = "OK " if ok else "x  "
            print(f"  {marca} {modelo}  -> {info}")
            if limite:
                print("  [429] free tier atingido; parando este provedor.")
                break

    print("\n" + "=" * 60)
    if testados == 0:
        print("Nenhum provedor testado.")
        if pulados:
            print("Os provedores selecionados foram pulados (sem chave ou sem Ollama).")
        else:
            print(
                "Configure LLM_API_KEY, GROQ_API_KEY, GEMINI_API_KEY ou "
                "OPENROUTER_API_KEY no .env."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

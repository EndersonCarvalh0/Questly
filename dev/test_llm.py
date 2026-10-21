"""Teste rápido: a NVIDIA devolve JSON válido de Questao?

Uso: .venv\\Scripts\\python.exe dev\\test_llm.py
"""

import sys

sys.path.insert(0, ".")

from pydantic import ValidationError

from src.llm import chat, extrair_json, get_client, get_modelo
from src.schema import Questao

TEXTO = (
    "A fotossíntese é o processo pelo qual as plantas convertem energia luminosa "
    "em energia química. Acontece nos cloroplastos, onde a clorofila absorve a luz "
    "solar. O gás carbônico da atmosfera e a água absorvida pelas raízes são "
    "transformados em glicose, e o oxigênio é liberado como subproduto."
)

PROMPT = f"""Você é um professor. Crie UMA questão de múltipla escolha sobre o texto abaixo.

Texto:
{TEXTO}

Regras:
- exatamente 4 alternativas distintas
- campo "correta" = índice (0-3) da alternativa correta
- campo "evidencia" = trecho literal do texto
- campo "pagina" = 1

Responda SOMENTE com JSON neste formato:
{{"enunciado": "...", "alternativas": ["...", "...", "...", "..."], "correta": 0, "justificativa": "...", "evidencia": "...", "pagina": 1}}
"""


def main() -> int:
    try:
        client = get_client()
    except RuntimeError as e:
        print(f"[ERRO] {e}")
        return 1

    print(f"Modelo: {get_modelo()}")
    try:
        modelos = [m.id for m in client.models.list()]
        print(f"Modelos disponíveis ({len(modelos)}): {', '.join(modelos[:15])}...")
    except Exception as e:
        print(f"[AVISO] não consegui listar modelos: {e}")

    print("\nChamando chat.completions...")
    try:
        saida = chat([{"role": "user", "content": PROMPT}], temperature=0.2)
    except Exception as e:
        print(f"[ERRO na chamada] {type(e).__name__}: {e}")
        return 1

    print(f"usage: {chat.ultimo_usage}")
    print(f"\nResposta bruta:\n{saida[:800]}\n")

    try:
        questao = Questao.model_validate(extrair_json(saida))
    except (ValueError, ValidationError) as e:
        print(f"[FALHA na validação] {e}")
        return 1

    print("[OK] Questao validada:")
    print(f"  enunciado : {questao.enunciado}")
    for i, alt in enumerate(questao.alternativas):
        mark = " <-- correta" if i == questao.correta else ""
        print(f"  {'ABCD'[i]}) {alt}{mark}")
    print(f"  evidencia : {questao.evidencia[:80]}...")
    print(f"  pagina    : {questao.pagina}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

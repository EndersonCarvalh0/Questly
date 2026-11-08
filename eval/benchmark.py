"""Benchmark simples do gerador: uma execução por versão de prompt.

Uso:
  .venv\\Scripts\\python.exe eval\\benchmark.py
  .venv\\Scripts\\python.exe eval\\benchmark.py --n 1 --versoes v2
  .venv\\Scripts\\python.exe eval\\benchmark.py --provedor nvidia --modelo openai/gpt-oss-20b

O benchmark usa o mesmo pipeline do app (src/pipeline.py). Ele não faz fallback
para provedor pago: se a API falhar, o erro é registrado e a execução continua
para a próxima versão.
"""

import argparse
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

from src.llm import get_modelo
from src.pipeline import gerar_simulado

PDF_PADRAO = Path("eval/pdfs/Aula01_ConceitosBasicos.pdf")
SAIDA_PADRAO = Path("eval/resultados.md")


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"\W+", " ", texto).strip()


def executar(pdf: Path, n: int, versao: str, provedor: str, modelo: str) -> dict:
    inicio = time.perf_counter()
    erro = None
    sim = None
    try:
        sim = gerar_simulado(
            pdf,
            n,
            versao=versao,
            provedor=provedor,
            modelo=modelo,
            etapa=lambda msg: print(f"  [{versao}] {msg}"),
        )
    except Exception as e:
        erro = f"{type(e).__name__}: {e}"
    tempo = time.perf_counter() - inicio
    corrigidas = 0
    duplicadas = 0
    if sim:
        corrigidas = sum("página corrigida" in linha for linha in sim.logs)
        vistos = set()
        for q in sim.questoes:
            chave = _normalizar(q.enunciado)
            if chave in vistos:
                duplicadas += 1
            else:
                vistos.add(chave)
    return {
        "versao": versao,
        "tempo": tempo,
        "erro": erro,
        "sim": sim,
        "corrigidas": corrigidas,
        "duplicadas": duplicadas,
    }


def formatar(pdf: Path, provedor: str, modelo: str, n: int, resultados: list[dict]) -> str:
    agora = datetime.now().strftime("%Y-%m-%d %H:%M")
    linhas = [
        "# Resultados do benchmark",
        "",
        f"- Data: {agora}",
        f"- PDF: `{pdf}`",
        f"- Provedor: `{provedor}`",
        f"- Modelo: `{modelo}`",
        f"- Questões solicitadas por versão: {n}",
        "",
        "| Versão | Válidas | Descartadas | Duplicadas | Página corrigida | Tempo (s) | Erro |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in resultados:
        sim = r["sim"]
        validas = len(sim.questoes) if sim else 0
        descartadas = sim.descartadas if sim else 0
        erro = r["erro"] or ""
        linhas.append(
            f"| {r['versao']} | {validas} | {descartadas} | {r['duplicadas']} | "
            f"{r['corrigidas']} | {r['tempo']:.1f} | {erro} |"
        )

    linhas.extend(["", "## Questões geradas", ""])
    for r in resultados:
        sim = r["sim"]
        linhas.append(f"### Prompt {r['versao']}")
        if not sim or not sim.questoes:
            linhas.append("Nenhuma questão válida.")
            linhas.append("")
            continue
        for i, q in enumerate(sim.questoes, start=1):
            linhas.append(f"{i}. **{q.enunciado}**")
            for j, alt in enumerate(q.alternativas):
                marca = " **(correta)**" if j == q.correta else ""
                linhas.append(f"   - {'ABCD'[j]}) {alt}{marca}")
            linhas.append(f"   - Página: {q.pagina}")
            linhas.append(f"   - Evidência: “{q.evidencia}”")
            linhas.append("")
        if sim.logs:
            linhas.append("Tentativas:")
            linhas.append("")
            for log in sim.logs:
                linhas.append(f"- {log}")
            linhas.append("")
    return "\n".join(linhas) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=PDF_PADRAO)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--versoes", default="v1,v2")
    parser.add_argument("--provedor", default="nvidia")
    parser.add_argument("--modelo", default=None)
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"[ERRO] PDF não encontrado: {args.pdf}")
        return 1

    try:
        modelo = args.modelo or get_modelo(args.provedor)
    except Exception as e:
        print(f"[ERRO] {e}")
        return 1
    versoes = [v.strip() for v in args.versoes.split(",") if v.strip()]
    if not versoes:
        print("[ERRO] nenhuma versão de prompt informada")
        return 1

    print(f"PDF: {args.pdf}")
    print(f"Provedor: {args.provedor}")
    print(f"Modelo: {modelo}")
    print(f"Versões: {', '.join(versoes)} | n={args.n}\n")

    resultados = []
    for versao in versoes:
        print(f"=== Prompt {versao} ===")
        resultados.append(executar(args.pdf, args.n, versao, args.provedor, modelo))
        r = resultados[-1]
        if r["erro"]:
            print(f"  erro: {r['erro']}")
        else:
            print(f"  tempo: {r['tempo']:.1f}s")
            print(f"  válidas: {len(r['sim'].questoes)} | descartadas: {r['sim'].descartadas}")

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(
        formatar(args.pdf, args.provedor, modelo, args.n, resultados),
        encoding="utf-8",
    )
    print(f"\nResultados salvos em {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import sys

sys.path.insert(0, ".")

from src.chunking import MODELO_PADRAO, chunk_por_paginas, get_model
from src.retrieval import Indice

PAGINAS = [
    # p.1 - fotossíntese
    "A fotossíntese converte energia luminosa em energia química. "
    "A clorofila nos cloroplastos absorve a luz solar. "
    "O gás carbônico e a água viram glicose e o oxigênio é liberado.",
    # p.2 - respiração celular
    "A respiração celular oxida a glicose para produzir ATP nas mitocôndrias. "
    "É um processo aeróbio que consome oxigênio e libera gás carbônico.",
    # p.3 - sistema solar
    "O Sistema Solar tem oito planetas. "
    "Mercúrio é o mais próximo do Sol e Netuno o mais distante. "
    "Júpiter é o maior planeta e possui uma grande mancha vermelha.",
]


def main() -> None:
    model = get_model(MODELO_PADRAO)
    chunks, _ = chunk_por_paginas(PAGINAS, max_tokens=250, sobreposicao=40)
    indice = Indice(model, chunks)
    print(f"{len(chunks)} chunks indexados\n")

    consultas = [
        "como as plantas produzem energia a partir da luz?",
        "qual planeta é o maior do sistema solar?",
        "onde ocorre a produção de ATP na célula?",
    ]
    for q in consultas:
        print(f"consulta: {q}")
        for t in indice.buscar(q, top_k=3):
            print(f"   {t.score:.3f}  {t.chunk.faixa_paginas}  {t.chunk.texto[:55]}...")
        print()


if __name__ == "__main__":
    main()

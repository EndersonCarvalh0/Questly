import sys

sys.path.insert(0, ".")

from src.chunking import Chunk
from src.schema import Questao
from src.validate import evidencia_no_contexto, normalizar, pagina_da_evidencia


def main() -> None:
    chunks = [
        Chunk(
            "A clorofila fica nos cloroplastos e absorve a luz solar.",
            [1],
            12,
        ),
        Chunk(
            "A respiração celular oxida a glicose para produzir ATP nas mitocôndrias.",
            [2],
            14,
        ),
    ]
    contexto = "\n".join(c.texto for c in chunks)
    questao = Questao(
        enunciado="Onde fica a clorofila que absorve a luz solar?",
        alternativas=["Mitocôndria", "Cloroplasto", "Ribossomo", "Lisossomo"],
        correta=1,
        justificativa="A clorofila está nos cloroplastos, onde ocorre a fotossíntese.",
        evidencia="A clorofila fica nos cloroplastos",
        pagina=2,
    )

    assert evidencia_no_contexto(questao, contexto)
    assert pagina_da_evidencia(questao.evidencia, chunks) == 1
    assert pagina_da_evidencia("texto que não existe", chunks) is None
    assert normalizar("A clorofila, nos cloroplastos!") == "a clorofila nos cloroplastos"
    print("[OK] validação determinística")


if __name__ == "__main__":
    main()

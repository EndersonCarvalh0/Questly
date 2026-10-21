import sys

sys.path.insert(0, ".")

from src.chunking import MODELO_PADRAO, chunk_por_paginas, contar_tokens, get_model

TEXTO = """A fotossíntese é o processo pelo qual as plantas convertem energia luminosa em energia química.
Ocorre nos cloroplastos, onde a clorofila absorve a luz solar.
O gás carbônico atmosférico e a água do solo são transformados em glicose.
O oxigênio é liberado como subproduto desse processo.
A respiração celular é o processo inverso, liberando energia a partir da glicose.
Nas mitocôndrias, a glicose é oxidada para produzir ATP, a moeda energética da célula.
A fermentação é uma via anaeróbia de obtenção de energia, comum em leveduras.
"""


def main() -> None:
    model = get_model(MODELO_PADRAO)
    print(f"modelo: {MODELO_PADRAO}  max_seq_length={model.max_seq_length}")
    print(f"tokens de 'passage: ' = {contar_tokens(model, '', 'passage: ')}")

    paginas = [TEXTO, TEXTO]
    chunks, avisos = chunk_por_paginas(paginas, max_tokens=250, sobreposicao=40)
    print(f"\n{len(chunks)} chunks:")
    for i, c in enumerate(chunks):
        print(f"  [{i}] {c.faixa_paginas} | {c.n_tokens} tokens | {c.texto[:60]}...")
    if avisos:
        print("\nAvisos:")
        for a in avisos:
            print(f"  - {a}")
    assert all(c.n_tokens <= model.max_seq_length for c in chunks)
    print("\n[OK] nenhum chunk acima de max_seq_length")


if __name__ == "__main__":
    main()

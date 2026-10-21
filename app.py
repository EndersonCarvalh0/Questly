import streamlit as st

from src.chunking import MODELO_PADRAO, chunk_por_paginas, get_model
from src.geracao import gerar_questoes
from src.ingest import extrair_paginas, validar_texto
from src.llm import get_modelo, listar_modelos_texto
from src.retrieval import Indice

st.set_page_config(page_title="QuestIA", page_icon="📄", layout="wide")

# Modelos liberados para a conta da equipe (models.list() mostra o catálogo
# inteiro, mas alguns dão 404). O usuário também pode carregar o catálogo.
MODELOS_CONHECIDOS = [
    "openai/gpt-oss-20b",
    "z-ai/glm-5.3",
    "z-ai/glm-5.3-flash",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "deepseek-ai/deepseek-v4.1-flash",
    "stepfun-ai/step-5-preview",
]


def init_state():
    for chave, valor in {
        "questoes": [],
        "trechos": [],
        "avisos": [],
        "logs": [],
        "respostas": {},
        "corrigido": False,
        "catalogo": None,
        "modelo_usado": None,
    }.items():
        st.session_state.setdefault(chave, valor)


def limpar_geracao():
    st.session_state.questoes = []
    st.session_state.trechos = []
    st.session_state.avisos = []
    st.session_state.logs = []
    st.session_state.respostas = {}
    st.session_state.corrigido = False


def render_questoes():
    questoes = st.session_state.questoes
    st.subheader(f"Simulado ({len(questoes)} questões)")
    for i, q in enumerate(questoes):
        st.markdown(f"**{i + 1}. {q.enunciado}**")
        opcoes = [f"{'ABCD'[j]}) {alt}" for j, alt in enumerate(q.alternativas)]
        escolha = st.radio(
            f"q{i}",
            opcoes,
            index=None,
            key=f"resp_{i}",
            label_visibility="collapsed",
        )
        st.session_state.respostas[i] = escolha

    if st.button("Corrigir", type="primary"):
        st.session_state.corrigido = True

    if st.session_state.corrigido:
        acertos = 0
        st.divider()
        for i, q in enumerate(questoes):
            escolha = st.session_state.respostas.get(i)
            idx = "ABCD".index(escolha[0]) if escolha else None
            ok = idx == q.correta
            acertos += int(ok)
            icone = "✅" if ok else "❌"
            st.markdown(
                f"{icone} **{i + 1}.** Gabarito: **{q.letra_correta})** "
                f"{q.alternativas[q.correta]}"
            )
            st.caption(f"Justificativa: {q.justificativa}")
            st.caption(f"Origem: {q.pagina} — evidência: “{q.evidencia}”")
        st.success(f"Nota: {acertos}/{len(questoes)}")


def aba_simulado():
    arquivo = st.file_uploader("Envie um PDF de aula (texto selecionável)", type="pdf")
    tema = st.text_input("Assunto (opcional)", placeholder="ex.: fotossíntese")

    opcoes = MODELOS_CONHECIDOS
    if st.session_state.catalogo:
        opcoes = st.session_state.catalogo
    padrao = get_modelo() if get_modelo() in opcoes else opcoes[0]
    col_m, col_b = st.columns([3, 1])
    modelo = col_m.selectbox("Modelo", opcoes, index=opcoes.index(padrao))
    if col_b.button("Carregar catálogo", help="Busca a lista completa da API"):
        try:
            with st.spinner("Consultando modelos..."):
                st.session_state.catalogo = listar_modelos_texto()
            st.rerun()
        except Exception as e:
            st.warning(f"Não consegui carregar o catálogo: {e}")

    col1, col2 = st.columns([1, 3])
    n = col1.number_input("Questões", min_value=1, max_value=5, value=3)
    gerar = col2.button("Gerar questões", type="primary", disabled=arquivo is None)

    if gerar and arquivo is not None:
        limpar_geracao()
        try:
            with st.status("Lendo o PDF...", expanded=True) as status:
                paginas = extrair_paginas(arquivo.getvalue())
                validar_texto(paginas)
                st.write(f"{len(paginas)} páginas lidas")

                status.update(label="Gerando embeddings e recuperando trechos...")
                model = get_model(MODELO_PADRAO)
                chunks, avisos = chunk_por_paginas(paginas, 250, 40)
                st.session_state.avisos = avisos
                if not chunks:
                    raise ValueError("Nenhum texto aproveitável foi extraído do PDF.")

                indice = Indice(model, chunks)
                if tema.strip():
                    trechos = indice.buscar(tema, top_k=min(5, len(chunks)))
                else:
                    trechos = indice.mais_central()[: min(5, len(chunks))]
                st.session_state.trechos = trechos
                st.write(f"{len(chunks)} chunks, {len(trechos)} trechos usados")

                status.update(label="Gerando e validando questões...")
                barra = st.progress(0.0)

                def progresso(i, total):
                    barra.progress((i + 1) / total)

                res = gerar_questoes(
                    [t.chunk for t in trechos], int(n), modelo=modelo, progresso=progresso
                )
                st.session_state.questoes = res.questoes
                st.session_state.logs = res.logs
                st.session_state.modelo_usado = modelo
                if res.descartadas:
                    st.warning(
                        f"{res.descartadas} questão(ões) descartada(s) após "
                        "falharem na validação."
                    )
                status.update(
                    label=f"{len(res.questoes)} questões prontas", state="complete"
                )
        except Exception as e:
            st.error(str(e))
            return

    if st.session_state.questoes:
        render_questoes()
        if st.button("Gerar de novo"):
            limpar_geracao()
            st.rerun()


def aba_como_funcionou():
    if st.session_state.get("modelo_usado"):
        st.caption(f"Modelo: `{st.session_state.modelo_usado}`")
    st.subheader("Trechos recuperados")
    trechos = st.session_state.trechos
    if not trechos:
        st.info("Gere um simulado para ver os trechos recuperados aqui.")
        return
    for t in trechos:
        st.markdown(
            f"**{t.chunk.faixa_paginas}** — score {t.score:.3f} — "
            f"{t.chunk.n_tokens} tokens"
        )
        st.caption(t.chunk.texto[:400] + ("..." if len(t.chunk.texto) > 400 else ""))

    st.divider()
    st.subheader("Validação")
    for linha in st.session_state.logs:
        st.text(linha)
    if st.session_state.avisos:
        st.subheader("Avisos de chunking")
        for a in st.session_state.avisos:
            st.text(a)


init_state()
st.title("QuestIA")
st.caption("Envie um PDF de aula e receba um simulado com gabarito, explicação e página de origem.")

aba1, aba2 = st.tabs(["Simulado", "Como funcionou"])
with aba1:
    aba_simulado()
with aba2:
    aba_como_funcionou()

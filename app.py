import json
import os
import queue
import threading
import time

import streamlit as st

from src.chunking import MODELO_PADRAO, contar_tokens, get_model
from src.llm import (
    get_client,
    get_modelo,
    listar_modelos_texto,
    ollama_disponivel,
    usa_ollama_local,
)
from src.pipeline import gerar_simulado
from src.providers import PROVEDORES, ids_modelos, lista_provedores, rotulo_modelo

st.set_page_config(page_title="QuestIA", page_icon="📄", layout="wide")

# Modelos liberados para a conta da equipe (models.list() mostra o catálogo
# inteiro, mas alguns dão 404). O usuário também pode carregar o catálogo.
MODELOS_CONHECIDOS = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "stepfun-ai/step-5-preview",
    "nvidia/nemotron-3.5-bolt-e4b",
    "mistralai/mistral-large-4-1t-a52b",
    "minimaxai/minimax-m3.1-flash",
]
MODELOS_OLLAMA = ["qwen2.5:3b", "qwen2.5:7b", "llama3.2:3b"]
STATUS_TESTE = {
    "ok": "OK",
    "falhou": "FALHOU",
    "corrigida": "CORRIGIDA",
    "nao_executado": "NÃO EXECUTADO",
}


def init_state():
    for chave, valor in {
        "questoes": [],
        "trechos": [],
        "avisos": [],
        "logs": [],
        "respostas": {},
        "corrigido": False,
        "catalogo": None,
        "catalogo_provedor": None,
        "modelo_usado": None,
        "provedor_usado": None,
        "n_paginas": 0,
        "n_chunks": 0,
        "geracao_id": 0,
        "geracao": None,
        "duracao": None,
        "cancelado": False,
        "erro_geracao": None,
        "descartadas": 0,
        "falhas": [],
        "checagens": [],
    }.items():
        st.session_state.setdefault(chave, valor)


def limpar_geracao():
    st.session_state.questoes = []
    st.session_state.trechos = []
    st.session_state.avisos = []
    st.session_state.logs = []
    st.session_state.respostas = {}
    st.session_state.corrigido = False
    st.session_state.modelo_usado = None
    st.session_state.provedor_usado = None
    st.session_state.n_paginas = 0
    st.session_state.n_chunks = 0
    st.session_state.geracao = None
    st.session_state.duracao = None
    st.session_state.cancelado = False
    st.session_state.erro_geracao = None
    st.session_state.descartadas = 0
    st.session_state.falhas = []
    st.session_state.checagens = []


def provedor_inicial() -> str:
    env = os.getenv("LLM_PROVIDER", "").strip().lower()
    if env in PROVEDORES:
        return env
    if usa_ollama_local() and ollama_disponivel():
        return "ollama"
    return "nvidia"


def provedores_para_ui() -> list[str]:
    env = os.getenv("LLM_PROVIDER", "").strip().lower()
    ollama_ok = ollama_disponivel()
    return [
        p
        for p in lista_provedores()
        if p != "ollama" or ollama_ok or env == "ollama"
    ]


def modelos_do_provedor(provedor: str) -> list[str]:
    if provedor == "nvidia":
        if usa_ollama_local("nvidia") and ollama_disponivel():
            return MODELOS_OLLAMA
        return MODELOS_CONHECIDOS
    if provedor == "ollama":
        return MODELOS_OLLAMA
    return ids_modelos(provedor)


def rotulo_provedor(provedor: str) -> str:
    p = PROVEDORES[provedor]
    sufixo = " (experimental)" if p.experimental else ""
    return f"{p.nome} — {p.melhor_para}{sufixo}"


def nome_provedor(provedor: str | None) -> str:
    if not provedor:
        return "NVIDIA (legado)"
    p = PROVEDORES.get(provedor)
    if not p:
        return provedor
    return f"{p.nome} (experimental)" if p.experimental else p.nome


def resumo_geracao() -> str:
    modelo = st.session_state.get("modelo_usado")
    if not modelo:
        return ""
    provedor = st.session_state.get("provedor_usado")
    return f"Provedor: {nome_provedor(provedor)} / Modelo: {modelo}"


def formatar_segundos(valor: float) -> str:
    return f"{valor:.1f} s".replace(".", ",")


def estado_tentativa(checks: list) -> str:
    falhas = [c for c in checks if c.status == "falhou"]
    if falhas:
        return f"falhou em {falhas[0].titulo}"
    if any(c.status == "corrigida" for c in checks):
        return "aprovada com correção de página"
    if any(c.status == "nao_executado" for c in checks):
        return "não concluída"
    return "aprovada"


def markdown_simulado() -> str:
    questoes = st.session_state.questoes
    linhas = ["# Simulado QuestIA", "", resumo_geracao(), ""]
    for i, q in enumerate(questoes):
        linhas.append(f"**{i + 1}. {q.enunciado}**")
        linhas.append("")
        for j, alt in enumerate(q.alternativas):
            linhas.append(f"- {'ABCD'[j]}) {alt}")
        linhas.append("")
        linhas.append(
            f"**Gabarito:** {q.letra_correta}) {q.alternativas[q.correta]}"
        )
        linhas.append(f"**Justificativa:** {q.justificativa}")
        linhas.append(f"**Origem:** p. {q.pagina} — evidência: “{q.evidencia}”")
        linhas.append("")
    return "\n".join(linhas)


def exportar_simulado():
    questoes = st.session_state.questoes
    if not questoes:
        return
    provedor = st.session_state.get("provedor_usado")
    dados = {
        "provedor": nome_provedor(provedor),
        "provedor_id": provedor,
        "modelo": st.session_state.get("modelo_usado"),
        "questoes": [q.model_dump() for q in questoes],
    }
    col_json, col_md = st.columns(2)
    col_json.download_button(
        "Baixar JSON",
        data=json.dumps(dados, ensure_ascii=False, indent=2),
        file_name="simulado.json",
        mime="application/json",
        key="download_json",
        use_container_width=True,
    )
    col_md.download_button(
        "Baixar Markdown",
        data=markdown_simulado(),
        file_name="simulado.md",
        mime="text/markdown",
        key="download_md",
        use_container_width=True,
    )


def render_questoes():
    questoes = st.session_state.questoes
    st.subheader(f"Simulado ({len(questoes)} questões)")
    st.caption(resumo_geracao())
    for i, q in enumerate(questoes):
        st.markdown(f"**{i + 1}. {q.enunciado}**")
        opcoes = [f"{'ABCD'[j]}) {alt}" for j, alt in enumerate(q.alternativas)]
        escolha = st.radio(
            f"q{i}",
            opcoes,
            index=None,
            key=f"resp_{st.session_state.geracao_id}_{i}",
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
            st.caption(f"Origem: p. {q.pagina} — evidência: “{q.evidencia}”")
        st.success(f"Nota: {acertos}/{len(questoes)}")

    st.divider()
    exportar_simulado()


def _finalizar_geracao(sim):
    st.session_state.questoes = sim.questoes
    st.session_state.trechos = sim.trechos
    st.session_state.avisos = sim.avisos
    st.session_state.logs = sim.logs
    st.session_state.n_paginas = sim.n_paginas
    st.session_state.n_chunks = sim.n_chunks
    st.session_state.modelo_usado = sim.modelo
    st.session_state.provedor_usado = sim.provedor
    st.session_state.duracao = sim.duracao
    st.session_state.cancelado = sim.cancelado
    st.session_state.erro_geracao = None
    st.session_state.descartadas = sim.descartadas
    st.session_state.falhas = [
        linha for linha in sim.logs if ": ok" not in linha
    ]
    st.session_state.checagens = sim.checagens
    st.session_state.geracao_id += 1
    st.session_state.geracao = None


def _finalizar_erro(mensagem: str, duracao: float):
    st.session_state.erro_geracao = mensagem
    st.session_state.duracao = duracao
    st.session_state.cancelado = False
    st.session_state.geracao = None


def _worker_geracao(job, arquivo_bytes, n, tema, modelo, provedor):
    fila = job["fila"]
    cancelar = job["cancelar"]
    inicio = time.perf_counter()
    try:
        sim = gerar_simulado(
            arquivo_bytes,
            n,
            tema=tema,
            modelo=modelo,
            provedor=provedor,
            progresso=lambda i, total: fila.put(("progresso", i, total)),
            etapa=lambda mensagem: fila.put(("etapa", mensagem)),
            cancelar=cancelar,
        )
        fila.put(("ok", sim))
    except Exception as e:
        fila.put(("erro", str(e), time.perf_counter() - inicio))


@st.fragment(run_every=0.5)
def _painel_geracao():
    job = st.session_state.get("geracao")
    if not job:
        return

    fila = job["fila"]
    while True:
        try:
            evento = fila.get_nowait()
        except queue.Empty:
            break
        tipo = evento[0]
        if tipo == "etapa":
            job["etapa"] = evento[1]
        elif tipo == "progresso":
            job["progresso"] = (evento[1] + 1) / evento[2]
        elif tipo == "ok":
            _finalizar_geracao(evento[1])
            st.rerun(scope="app")
            return
        elif tipo == "erro":
            _finalizar_erro(evento[1], evento[2])
            st.rerun(scope="app")
            return

    st.progress(job.get("progresso", 0.0))
    st.caption(job.get("etapa", "Iniciando..."))
    if st.button(
        "Parar geração",
        disabled=job.get("parando", False),
        key=f"parar_geracao_{job['id']}",
    ):
        job["cancelar"].set()
        job["parando"] = True
    if job.get("parando"):
        st.info("Parando após a questão atual...")


def aba_simulado():
    arquivo = st.file_uploader("Envie um PDF de aula (texto selecionável)", type="pdf")
    tema = st.text_input("Assunto (opcional)", placeholder="ex.: fotossíntese")

    provedores = provedores_para_ui()
    if not provedores:
        st.error("Nenhum provedor disponível. Configure uma chave no .env.")
        return

    inicial = provedor_inicial()
    if inicial not in provedores:
        inicial = provedores[0]
    col_p, col_lim = st.columns([3, 1])
    provedor = col_p.selectbox(
        "Provedor",
        provedores,
        index=provedores.index(inicial),
        format_func=rotulo_provedor,
    )
    col_lim.caption(PROVEDORES[provedor].limite)
    if PROVEDORES[provedor].experimental:
        st.caption(
            "Provedor experimental: use a NVIDIA como caminho principal da demo."
        )
    if provedor == "ollama" and not ollama_disponivel():
        st.caption("Ollama não detectado. Instale-o ou escolha outro provedor.")

    modo_ollama = provedor == "ollama" or (
        provedor == "nvidia" and usa_ollama_local("nvidia") and ollama_disponivel()
    )
    catalogo_ativo = bool(
        st.session_state.catalogo_provedor == provedor
        and st.session_state.catalogo
        and not modo_ollama
    )
    opcoes = (
        st.session_state.catalogo
        if catalogo_ativo
        else modelos_do_provedor(provedor)
    )
    try:
        padrao = get_modelo(provedor)
    except Exception as e:
        padrao = opcoes[0] if opcoes else ""
        st.warning(str(e))
    if padrao and padrao not in opcoes:
        opcoes = [padrao] + opcoes
    if catalogo_ativo:
        st.caption(
            "Catálogo completo da API; alguns modelos podem não estar liberados."
        )
    col_m, col_b = st.columns([3, 1])
    modelo = col_m.selectbox(
        "Modelo",
        opcoes,
        index=opcoes.index(padrao) if padrao in opcoes else 0,
        format_func=lambda m: rotulo_modelo(provedor, m),
    )
    rotulo_catalogo = (
        "Usar modelos conhecidos"
        if catalogo_ativo
        else "Carregar catálogo completo"
    )
    ajuda_catalogo = (
        "Volta para a lista de modelos que sabemos que funcionam."
        if catalogo_ativo
        else "Busca a lista completa da API; alguns modelos podem falhar."
    )
    if col_b.button(rotulo_catalogo, help=ajuda_catalogo):
        if catalogo_ativo:
            st.session_state.catalogo = None
            st.session_state.catalogo_provedor = None
            st.rerun()
        try:
            with st.spinner("Consultando modelos..."):
                st.session_state.catalogo = listar_modelos_texto(provedor)
                st.session_state.catalogo_provedor = provedor
            st.rerun()
        except Exception as e:
            st.warning(f"Não consegui carregar o catálogo: {e}")

    try:
        get_client(provedor)
        erro_conexao = None
    except Exception as e:
        erro_conexao = str(e)
    if erro_conexao:
        st.warning(erro_conexao)

    col1, col2 = st.columns([1, 3])
    n = col1.number_input("Questões", min_value=1, max_value=5, value=3)
    gerando = st.session_state.get("geracao") is not None
    gerar = col2.button(
        "Gerar questões",
        type="primary",
        disabled=arquivo is None or erro_conexao is not None or gerando,
    )

    if gerar and arquivo is not None:
        limpar_geracao()
        job = {
            "id": time.time_ns(),
            "fila": queue.Queue(),
            "cancelar": threading.Event(),
            "parando": False,
            "etapa": "Iniciando...",
            "progresso": 0.0,
        }
        st.session_state.geracao = job
        threading.Thread(
            target=_worker_geracao,
            args=(job, arquivo.getvalue(), int(n), tema, modelo, provedor),
            daemon=True,
        ).start()
        st.rerun(scope="app")

    if st.session_state.get("geracao"):
        _painel_geracao()

    if st.session_state.get("erro_geracao"):
        st.error(st.session_state.erro_geracao)

    if st.session_state.get("falhas"):
        with st.expander("Detalhes das tentativas"):
            for linha in st.session_state.falhas:
                st.code(linha)
    if st.session_state.get("descartadas"):
        st.warning(
            f"{st.session_state.descartadas} questão(ões) descartada(s) após "
            "falharem na validação."
        )

    if st.session_state.questoes:
        render_questoes()
        if st.button("Gerar de novo"):
            limpar_geracao()
            st.rerun()


def aba_como_funcionou():
    if st.session_state.get("modelo_usado"):
        st.caption(resumo_geracao())
    duracao = st.session_state.get("duracao")
    if duracao is not None and st.session_state.get("modelo_usado"):
        if st.session_state.get("cancelado"):
            st.warning(
                "Geração interrompida após "
                f"{len(st.session_state.questoes)} questão(ões)."
            )
            st.metric("Tempo até a interrupção", formatar_segundos(duracao))
        else:
            st.metric("Tempo de geração", formatar_segundos(duracao))
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
    st.subheader("Testes executados")
    st.caption(
        "Cada tentativa passa pelos testes abaixo. O detalhe aparece apenas "
        "quando o teste falha ou corrige a página."
    )
    st.table(
        [
            {
                "Teste": "Chamada ao LLM",
                "O que verifica": "Se a API respondeu sem erro.",
            },
            {
                "Teste": "Formato JSON",
                "O que verifica": "Se a resposta contém um objeto JSON válido.",
            },
            {
                "Teste": "Estrutura, alternativas e índice",
                "O que verifica": (
                    "Campos obrigatórios, 4 alternativas distintas e índice "
                    "da correta entre 0 e 3."
                ),
            },
            {
                "Teste": "Evidência no contexto",
                "O que verifica": (
                    "Se a evidência aparece literalmente no contexto recuperado."
                ),
            },
            {
                "Teste": "Página da evidência",
                "O que verifica": (
                    "Se a página real do trecho confere; corrige quando necessário."
                ),
            },
        ]
    )
    checagens = st.session_state.get("checagens") or []
    if not checagens:
        st.info("Gere um simulado para ver os testes executados.")
    else:
        grupos: dict[tuple[int, int], list] = {}
        for c in checagens:
            grupos.setdefault((c.questao, c.tentativa), []).append(c)
        for (questao, tentativa), checks in grupos.items():
            estado = estado_tentativa(checks)
            with st.expander(
                f"Q{questao} · Tentativa {tentativa} — {estado}"
            ):
                for c in checks:
                    status = STATUS_TESTE.get(c.status, c.status)
                    linha = f"**{c.titulo}** — {status}"
                    if c.detalhe and c.status != "ok":
                        linha += f": {c.detalhe}"
                    st.markdown(linha)
    if st.session_state.logs:
        with st.expander("Logs técnicos"):
            for linha in st.session_state.logs:
                st.text(linha)
    if st.session_state.avisos:
        st.subheader("Avisos de chunking")
        for a in st.session_state.avisos:
            st.text(a)


def aba_fundamentos():
    st.subheader("Os 5 conteúdos da unidade no QuestIA")
    n_paginas = st.session_state.get("n_paginas", 0)
    n_chunks = st.session_state.get("n_chunks", 0)
    trechos = st.session_state.get("trechos", [])
    if n_paginas:
        st.caption(
            f"Último processamento: {n_paginas} páginas, {n_chunks} chunks, "
            f"{len(trechos)} trechos recuperados."
        )

    st.markdown(
        "**1. Tokenização.** O PDF é extraído página a página, dividido em "
        "frases e contado em tokens pelo tokenizer do modelo de embeddings. "
        "Cada chunk respeita o limite de 512 tokens do `e5-small`."
    )
    st.markdown(
        "**2. Embeddings.** Cada chunk vira um vetor com o modelo local "
        "`intfloat/multilingual-e5-small`. Usamos os prefixos `passage:` e "
        "`query:`, normalizamos os vetores e medimos similaridade por cosseno."
    )
    st.markdown(
        "**3. Transformer.** O encoder transforma tokens em vetores (é o que o "
        "e5 faz); o decoder gera texto token a token (é o que o LLM faz). A "
        "atenção permite que cada token pese o contexto inteiro."
    )
    st.markdown(
        "**4. Engenharia de prompt.** O prompt v2 pede JSON estrito, 4 "
        "alternativas, evidência literal e página. Se a validação falha, uma "
        "nova tentativa recebe o motivo da rejeição."
    )
    st.markdown(
        "**5. RAG.** O fluxo é: PDF → extração → chunking → embeddings → "
        "recuperação top-k → prompt com os trechos → LLM → validação "
        "determinística. O modelo não recebe o PDF inteiro, só o contexto "
        "recuperado."
    )

    with st.expander("Ver a tokenização de um exemplo"):
        texto = st.text_area(
            "Texto",
            "A fotossíntese ocorre nos cloroplastos, onde a clorofila absorve a luz.",
            key="fund_texto",
        )
        if st.button("Contar tokens", key="fund_botao"):
            model = get_model(MODELO_PADRAO)
            st.write(f"`{contar_tokens(model, texto)}` tokens em `{MODELO_PADRAO}`")
            st.code(" ".join(model.tokenizer.tokenize(texto)))

    if trechos:
        st.markdown("**Scores da última recuperação**")
        for t in trechos:
            st.write(f"{t.chunk.faixa_paginas} — {t.score:.3f}")

    st.markdown("### Onde cada conteúdo aparece no código")
    st.table(
        [
            {"Conteúdo": "Tokenização", "Arquivo": "src/chunking.py"},
            {"Conteúdo": "Embeddings", "Arquivo": "src/embeddings.py"},
            {"Conteúdo": "Transformer", "Arquivo": "src/llm.py"},
            {"Conteúdo": "Engenharia de prompt", "Arquivo": "src/prompts.py"},
            {"Conteúdo": "RAG", "Arquivo": "src/retrieval.py e src/pipeline.py"},
            {"Conteúdo": "Validação", "Arquivo": "src/validate.py"},
        ]
    )


init_state()
st.title("QuestIA")
st.caption(
    "Envie um PDF de aula e receba um simulado com gabarito, explicação e "
    "página de origem."
)

aba1, aba2, aba3 = st.tabs(["Simulado", "Como funcionou", "Fundamentos"])
with aba1:
    aba_simulado()
with aba2:
    aba_como_funcionou()
with aba3:
    aba_fundamentos()

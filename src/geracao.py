from dataclasses import dataclass, field
import random
from threading import Event

from pydantic import ValidationError

from src.chunking import Chunk
from src.llm import chat, extrair_json
from src.prompts import montar_contexto, prompt_v1, prompt_v2
from src.schema import Questao
from src.validate import evidencia_no_contexto, pagina_da_evidencia

PROMPTS = {"v1": prompt_v1, "v2": prompt_v2}

PROVEDORES_GRATUITOS = ("nvidia", "openrouter", "groq", "gemini")

CHECAGEM_LLM = "Chamada ao LLM"
CHECAGEM_JSON = "Formato JSON"
CHECAGEM_ESTRUTURA = "Estrutura, alternativas e índice"
CHECAGEM_EVIDENCIA = "Evidência no contexto"
CHECAGEM_PAGINA = "Página da evidência"


def _limite_de_uso(erro: Exception) -> bool:
    status = getattr(erro, "status_code", None)
    texto = str(erro).lower()
    return status == 429 or "429" in texto or "resource_exhausted" in texto


def _erro_fatal(erro: Exception) -> bool:
    status = getattr(erro, "status_code", None)
    if status in (401, 403, 404):
        return True
    texto = str(erro).lower()
    return any(
        chave in texto
        for chave in (
            "401",
            "403",
            "404",
            "unauthorized",
            "forbidden",
            "function not found for account",
            "model not found",
        )
    )


def _erro_configuracao(erro: Exception) -> bool:
    if not isinstance(erro, RuntimeError):
        return False
    texto = str(erro).lower()
    return (
        "configure" in texto
        or "não está disponível" in texto
        or "não está instalado" in texto
        or ("chave" in texto and "inválida" in texto)
    )


def _mensagem_fatal(erro: Exception, provedor: str | None) -> str:
    status = getattr(erro, "status_code", None)
    nome = provedor or "NVIDIA"
    if status == 401:
        return (
            f"Chave inválida ou ausente (401) em {nome}. "
            "Confira a chave configurada no .env."
        )
    if status == 403:
        return (
            f"Acesso negado (403) em {nome}. "
            "Confira se a chave tem permissão para o modelo escolhido."
        )
    if status == 404:
        return (
            f"Modelo não encontrado ou não liberado (404) em {nome}. "
            "Escolha outro modelo na interface."
        )
    return f"Erro fatal em {nome}: {erro}"


@dataclass
class Checagem:
    questao: int
    tentativa: int
    titulo: str
    status: str
    detalhe: str = ""


@dataclass
class Resultado:
    questoes: list[Questao] = field(default_factory=list)
    descartadas: int = 0
    usages: list[dict] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    cancelado: bool = False
    checagens: list[Checagem] = field(default_factory=list)


def _gerar_uma(
    contexto: str,
    versao: str,
    temperatura: float,
    evita: list[str],
    modelo: str | None = None,
    feedback: str | None = None,
    provedor: str | None = None,
    trechos: list[Chunk] | None = None,
    checagens: list[tuple[str, str, str]] | None = None,
) -> tuple[Questao | None, str]:
    def registrar(titulo: str, status: str, detalhe: str = "") -> None:
        if checagens is not None:
            checagens.append((titulo, status, detalhe))

    def pular(*titulos: str) -> None:
        for titulo in titulos:
            registrar(titulo, "nao_executado")

    mensagens = PROMPTS[versao](contexto, evita)
    if feedback:
        mensagens.append(
            {
                "role": "user",
                "content": (
                    "CORRIJA A RESPOSTA ANTERIOR. Ela foi rejeitada por este motivo: "
                    f"{feedback}. Gere novamente um único objeto JSON válido com "
                    "todos os campos obrigatórios: enunciado, alternativas (4), "
                    "correta (índice 0 a 3), justificativa, evidencia e pagina. "
                    "A evidencia deve ser uma sequência de palavras copiada do material, "
                    "sem reticências nem paráfrase. Responda somente com o JSON."
                ),
            }
        )
    try:
        bruto = chat(
            mensagens, temperature=temperatura, modelo=modelo, provedor=provedor
        )
    except Exception as e:
        if _erro_configuracao(e):
            raise
        if provedor in PROVEDORES_GRATUITOS and _limite_de_uso(e):
            raise RuntimeError(
                "Limite do plano gratuito atingido (429). "
                "Aguarde o limite renovar ou escolha outro provedor/modelo."
            ) from e
        if _erro_fatal(e):
            raise RuntimeError(_mensagem_fatal(e, provedor)) from e
        registrar(CHECAGEM_LLM, "falhou", type(e).__name__)
        pular(
            CHECAGEM_JSON,
            CHECAGEM_ESTRUTURA,
            CHECAGEM_EVIDENCIA,
            CHECAGEM_PAGINA,
        )
        return None, f"erro na API: {type(e).__name__}: {e}"
    registrar(CHECAGEM_LLM, "ok")
    try:
        dados = extrair_json(bruto)
    except ValueError as e:
        registrar(CHECAGEM_JSON, "falhou", "JSON inválido ou ausente")
        pular(CHECAGEM_ESTRUTURA, CHECAGEM_EVIDENCIA, CHECAGEM_PAGINA)
        return None, f"JSON inválido: {e}"
    registrar(CHECAGEM_JSON, "ok")
    try:
        questao = Questao.model_validate(dados)
    except ValidationError as e:
        registrar(CHECAGEM_ESTRUTURA, "falhou", "campos ou alternativas inválidas")
        pular(CHECAGEM_EVIDENCIA, CHECAGEM_PAGINA)
        return None, f"JSON inválido: {e}"
    registrar(CHECAGEM_ESTRUTURA, "ok")
    if not evidencia_no_contexto(questao, contexto):
        registrar(CHECAGEM_EVIDENCIA, "falhou", "trecho não encontrado no contexto")
        pular(CHECAGEM_PAGINA)
        return None, "evidência não encontrada no contexto"
    registrar(CHECAGEM_EVIDENCIA, "ok")
    if trechos:
        pagina = pagina_da_evidencia(questao.evidencia, trechos)
        if pagina is None:
            registrar(CHECAGEM_PAGINA, "falhou", "evidência não encontrada nos trechos")
            return None, "evidência não encontrada nos trechos recuperados"
        if questao.pagina != pagina:
            original = questao.pagina
            questao = questao.model_copy(update={"pagina": pagina})
            registrar(CHECAGEM_PAGINA, "corrigida", f"de {original} para {pagina}")
            return questao, f"ok (página corrigida de {original} para {pagina})"
        registrar(CHECAGEM_PAGINA, "ok")
    else:
        registrar(CHECAGEM_PAGINA, "nao_executado", "sem trechos recuperados")
    return questao, "ok"


def gerar_questoes(
    trechos: list[Chunk],
    n: int,
    versao: str = "v2",
    temperatura: float = 0.2,
    tentativas: int = 2,
    modelo: str | None = None,
    progresso=None,
    provedor: str | None = None,
    cancelar: Event | None = None,
) -> Resultado:
    contexto = montar_contexto(trechos)
    res = Resultado()
    evita: list[str] = []
    # Distribui a posição correta de forma equilibrada no simulado. O LLM
    # costuma favorecer uma posição; embaralhamos após validar a questão para
    # evitar que esse viés apareça no gabarito apresentado ao usuário.
    posicoes_corretas = list(range(4)) * ((n + 3) // 4)
    random.shuffle(posicoes_corretas)

    for i in range(n):
        if cancelar and cancelar.is_set():
            res.cancelado = True
            break
        if progresso:
            progresso(i, n)
        questao = None
        feedback = None
        for t in range(tentativas):
            checagens_tentativa: list[tuple[str, str, str]] = []
            questao, motivo = _gerar_uma(
                contexto,
                versao,
                temperatura,
                evita,
                modelo,
                feedback,
                provedor,
                trechos,
                checagens_tentativa,
            )
            for titulo, status, detalhe in checagens_tentativa:
                res.checagens.append(
                    Checagem(i + 1, t + 1, titulo, status, detalhe)
                )
            if t == 0:
                res.usages.append(dict(chat.ultimo_usage))
            res.logs.append(f"Q{i+1} tentativa {t+1}: {motivo}")
            if questao:
                break
            if cancelar and cancelar.is_set():
                break
            feedback = motivo
        if questao:
            posicao_correta = posicoes_corretas[i]
            alternativas = questao.alternativas.copy()
            resposta_correta = alternativas.pop(questao.correta)
            random.shuffle(alternativas)
            alternativas.insert(posicao_correta, resposta_correta)
            questao = questao.model_copy(
                update={"alternativas": alternativas, "correta": posicao_correta}
            )
            res.questoes.append(questao)
            evita.append(questao.enunciado)
        else:
            res.descartadas += 1
        if cancelar and cancelar.is_set():
            res.cancelado = True
            break

    return res

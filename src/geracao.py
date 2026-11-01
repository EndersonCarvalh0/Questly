from dataclasses import dataclass, field
import random

from pydantic import ValidationError

from src.chunking import Chunk
from src.llm import chat, extrair_json
from src.prompts import montar_contexto, prompt_v1, prompt_v2
from src.schema import Questao
from src.validate import evidencia_no_contexto

PROMPTS = {"v1": prompt_v1, "v2": prompt_v2}


@dataclass
class Resultado:
    questoes: list[Questao] = field(default_factory=list)
    descartadas: int = 0
    usages: list[dict] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


def _gerar_uma(
    contexto: str,
    versao: str,
    temperatura: float,
    evita: list[str],
    modelo: str | None = None,
    feedback: str | None = None,
) -> tuple[Questao | None, str]:
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
        bruto = chat(mensagens, temperature=temperatura, modelo=modelo)
    except Exception as e:
        return None, f"erro na API: {type(e).__name__}: {e}"
    try:
        dados = extrair_json(bruto)
        questao = Questao.model_validate(dados)
    except (ValueError, ValidationError) as e:
        return None, f"JSON inválido: {e}"
    if not evidencia_no_contexto(questao, contexto):
        return None, "evidência não encontrada no contexto"
    return questao, "ok"


def gerar_questoes(
    trechos: list[Chunk],
    n: int,
    versao: str = "v2",
    temperatura: float = 0.2,
    tentativas: int = 2,
    modelo: str | None = None,
    progresso=None,
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
        if progresso:
            progresso(i, n)
        questao = None
        feedback = None
        for t in range(tentativas):
            questao, motivo = _gerar_uma(
                contexto, versao, temperatura, evita, modelo, feedback
            )
            if t == 0:
                res.usages.append(dict(chat.ultimo_usage))
            res.logs.append(f"Q{i+1} tentativa {t+1}: {motivo}")
            if questao:
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

    return res

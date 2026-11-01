import json
import os
import re

from dotenv import load_dotenv
from openai import APIConnectionError, OpenAI

load_dotenv()

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
OLLAMA_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:3b"


def usa_ollama_local() -> bool:
    """Sem chave NVIDIA, usa Ollama local automaticamente."""
    chave = os.getenv("LLM_API_KEY", "").strip()
    return not chave or chave.startswith("nvapi-...")


def get_client() -> OpenAI:
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    if usa_ollama_local():
        return OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", OLLAMA_URL),
            api_key="ollama",
            max_retries=2,
            timeout=300,
        )
    if not base_url:
        raise RuntimeError(
            "Configure LLM_BASE_URL no arquivo .env."
        )
    return OpenAI(base_url=base_url, api_key=api_key, max_retries=3, timeout=120)


def get_modelo() -> str:
    if usa_ollama_local():
        return os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
    modelo = os.getenv("LLM_MODEL")
    if not modelo:
        raise RuntimeError("LLM_MODEL não definido no .env")
    return modelo


def listar_modelos_texto() -> list[str]:
    """Modelos do catálogo, sem os que claramente não são de chat/texto."""
    excluir = (
        "embed", "vision", "deplot", "neva", "vila", "nvclip", "kosmos",
        "fuyu", "guard", "safety", "content-safety", "topic-control",
        "reward", "code", "video-detector", "parse", "diffusion",
        "recurrentgemma", "starcoder", "riva-translate",
    )
    ids = sorted(m.id for m in get_client().models.list())
    return [i for i in ids if not any(e in i for e in excluir)]


def _parametros_extras(modelo: str) -> dict:
    extras: dict = {}
    thinking = os.getenv("LLM_THINKING", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if "nemotron-3-ultra" in modelo.lower():
        extras["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": thinking}
        }
        extras["max_tokens"] = int(os.getenv("LLM_MAX_TOKENS", "16384"))
    elif os.getenv("LLM_MAX_TOKENS") and not usa_ollama_local():
        extras["max_tokens"] = int(os.getenv("LLM_MAX_TOKENS"))
    return extras


def chat(mensagens: list[dict], temperature: float = 0.2, modelo: str | None = None) -> str:
    client = get_client()
    modelo_uso = modelo or get_modelo()
    try:
        resp = client.chat.completions.create(
            model=modelo_uso,
            messages=mensagens,
            temperature=temperature,
            **_parametros_extras(modelo_uso),
        )
    except APIConnectionError as e:
        if usa_ollama_local():
            raise RuntimeError(
                "Não consegui conectar ao Ollama em http://localhost:11434. "
                "Abra o Ollama e baixe o modelo com: ollama pull qwen2.5:3b."
            ) from e
        raise
    usage = resp.usage
    if usage:
        chat.ultimo_usage = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    else:
        chat.ultimo_usage = {}
    escolha = resp.choices[0]
    conteudo = escolha.message.content or ""
    if not conteudo.strip():
        motivo = getattr(escolha, "finish_reason", None) or "desconhecido"
        raise RuntimeError(
            "O modelo retornou conteúdo vazio "
            f"(finish_reason={motivo}). Verifique o limite de tokens e tente novamente."
        )
    return conteudo


chat.ultimo_usage = {}


def extrair_json(texto: str) -> dict:
    m = _FENCE.search(texto)
    if m:
        texto = m.group(1)
    inicio, fim = texto.find("{"), texto.rfind("}")
    if inicio == -1 or fim == -1 or fim < inicio:
        raise ValueError("nenhum objeto JSON encontrado na resposta")
    return json.loads(texto[inicio : fim + 1])

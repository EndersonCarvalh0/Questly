import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def get_client() -> OpenAI:
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    if not base_url or not api_key or api_key.startswith("nvapi-..."):
        raise RuntimeError(
            "Configure o arquivo .env (copie .env.example e preencha LLM_API_KEY)."
        )
    return OpenAI(base_url=base_url, api_key=api_key, max_retries=3, timeout=120)


def get_modelo() -> str:
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


def chat(mensagens: list[dict], temperature: float = 0.2, modelo: str | None = None) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=modelo or get_modelo(),
        messages=mensagens,
        temperature=temperature,
    )
    usage = resp.usage
    if usage:
        chat.ultimo_usage = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    else:
        chat.ultimo_usage = {}
    return resp.choices[0].message.content or ""


chat.ultimo_usage = {}


def extrair_json(texto: str) -> dict:
    m = _FENCE.search(texto)
    if m:
        texto = m.group(1)
    inicio, fim = texto.find("{"), texto.rfind("}")
    if inicio == -1 or fim == -1 or fim < inicio:
        raise ValueError("nenhum objeto JSON encontrado na resposta")
    return json.loads(texto[inicio : fim + 1])

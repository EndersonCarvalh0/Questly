import json
import os
import re
import time
import urllib.request

from dotenv import load_dotenv
from openai import APIConnectionError, OpenAI

from src.providers import obter_provedor

load_dotenv()

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
OLLAMA_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:3b"
_OLLAMA_STATUS = {"ts": 0.0, "ok": False}
_OLLAMA_TTL = 15.0

_EXCLUIR = (
    "embed", "vision", "deplot", "neva", "vila", "nvclip", "kosmos",
    "fuyu", "guard", "safety", "content-safety", "topic-control",
    "reward", "code", "video-detector", "parse", "diffusion",
    "recurrentgemma", "starcoder", "riva-translate",
)


def _provedor_efetivo(provedor: str | None = None) -> str | None:
    if provedor:
        return provedor.strip().lower()
    env = os.getenv("LLM_PROVIDER", "").strip().lower()
    return env or None


def ollama_disponivel(timeout: float = 0.5) -> bool:
    """Checagem curta e cacheada do servidor local do Ollama."""
    agora = time.monotonic()
    if agora - _OLLAMA_STATUS["ts"] < _OLLAMA_TTL:
        return bool(_OLLAMA_STATUS["ok"])

    base = os.getenv("OLLAMA_BASE_URL", OLLAMA_URL).rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout) as resp:
            ok = 200 <= getattr(resp, "status", 0) < 300
    except Exception:
        ok = False

    _OLLAMA_STATUS["ts"] = agora
    _OLLAMA_STATUS["ok"] = ok
    return ok


def usa_ollama_local(provedor: str | None = None) -> bool:
    """Sem chave NVIDIA, usa Ollama local automaticamente."""
    prov = _provedor_efetivo(provedor)
    if prov and prov != "nvidia":
        return prov == "ollama"
    chave = os.getenv("LLM_API_KEY", "").strip()
    return not chave or chave.startswith("nvapi-...")


def _client_legado(provedor: str | None = None) -> OpenAI:
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    if usa_ollama_local(provedor):
        if not ollama_disponivel():
            raise RuntimeError(
                "LLM_API_KEY não configurada e o Ollama não está disponível. "
                "Configure a chave da NVIDIA no .env ou escolha outro provedor."
            )
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


def _client_do_provedor(provedor: str) -> OpenAI:
    p = obter_provedor(provedor)
    if p.id == "nvidia":
        return _client_legado(p.id)
    if p.id == "ollama":
        if not ollama_disponivel():
            raise RuntimeError(
                "O Ollama não está instalado/executando em http://localhost:11434. "
                "Instale o Ollama ou escolha outro provedor."
            )
        return OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", p.base_url),
            api_key="ollama",
            max_retries=2,
            timeout=300,
        )
    chave = os.getenv(p.env_key, "").strip()
    if not chave or chave.endswith("..."):
        raise RuntimeError(
            f"Configure {p.env_key} no arquivo .env para usar {p.nome}."
        )
    return OpenAI(base_url=p.base_url, api_key=chave, max_retries=3, timeout=120)


def get_client(provedor: str | None = None) -> OpenAI:
    prov = _provedor_efetivo(provedor)
    if prov is None or prov == "nvidia":
        return _client_legado(prov)
    return _client_do_provedor(prov)


def get_modelo(provedor: str | None = None) -> str:
    prov = _provedor_efetivo(provedor)
    if prov is None or prov == "nvidia":
        if usa_ollama_local(prov) and ollama_disponivel():
            return os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
        modelo = os.getenv("LLM_MODEL")
        if not modelo:
            raise RuntimeError("LLM_MODEL não definido no .env")
        return modelo
    p = obter_provedor(prov)
    override = os.getenv(p.env_modelo, "").strip()
    if override:
        return override
    if p.id == "ollama":
        return os.getenv("OLLAMA_MODEL", p.modelos[0].id)
    return p.modelos[0].id


def listar_modelos_texto(provedor: str | None = None) -> list[str]:
    """Modelos do catálogo, sem os que claramente não são de chat/texto."""
    prov = _provedor_efetivo(provedor)
    if prov is None or prov == "nvidia":
        ids = sorted(m.id for m in get_client(prov).models.list())
        return [i for i in ids if not any(e in i for e in _EXCLUIR)]
    p = obter_provedor(prov)
    try:
        ids = sorted(m.id for m in get_client(p.id).models.list())
    except Exception:
        return [m.id for m in p.modelos]
    if p.somente_gratis:
        return [i for i in ids if i.endswith(":free")]
    return [i for i in ids if not any(e in i for e in _EXCLUIR)]


def _parametros_extras(modelo: str, provedor: str | None = None) -> dict:
    prov = _provedor_efetivo(provedor)
    extras: dict = {}
    if prov is None or prov == "nvidia":
        thinking = os.getenv("LLM_THINKING", "false").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if "nemotron-3-ultra" in modelo.lower():
            extras["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": thinking}
            }
            extras["max_tokens"] = int(os.getenv("LLM_MAX_TOKENS", "16384"))
        elif os.getenv("LLM_MAX_TOKENS") and not usa_ollama_local(prov):
            extras["max_tokens"] = int(os.getenv("LLM_MAX_TOKENS"))
        return extras
    if prov == "openrouter":
        if "reasoning" in modelo.lower() or "ultra" in modelo.lower():
            esforco = os.getenv("OPENROUTER_REASONING_EFFORT", "low").strip().lower()
            if esforco and esforco != "none":
                extras["extra_body"] = {"reasoning": {"effort": esforco}}
        extras["max_tokens"] = int(
            os.getenv("OPENROUTER_MAX_TOKENS", os.getenv("LLM_MAX_TOKENS", "8192"))
        )
    return extras


def chat(
    mensagens: list[dict],
    temperature: float = 0.2,
    modelo: str | None = None,
    provedor: str | None = None,
) -> str:
    chat.ultimo_usage = {}
    client = get_client(provedor)
    modelo_uso = modelo or get_modelo(provedor)
    try:
        resp = client.chat.completions.create(
            model=modelo_uso,
            messages=mensagens,
            temperature=temperature,
            **_parametros_extras(modelo_uso, provedor),
        )
    except APIConnectionError as e:
        if usa_ollama_local(provedor):
            if ollama_disponivel():
                raise RuntimeError(
                    "Não consegui conectar ao Ollama em http://localhost:11434. "
                    "Verifique se o servidor está em execução e se o modelo foi baixado."
                ) from e
            raise RuntimeError(
                "LLM_API_KEY não configurada e o Ollama não está disponível. "
                "Configure a chave da NVIDIA no .env ou escolha outro provedor."
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

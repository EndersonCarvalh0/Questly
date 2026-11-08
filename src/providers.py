from dataclasses import dataclass


@dataclass(frozen=True)
class Modelo:
    id: str
    apelido: str = ""
    melhor_para: str = ""


@dataclass(frozen=True)
class Provedor:
    id: str
    nome: str
    base_url: str
    env_key: str
    env_modelo: str
    modelos: tuple[Modelo, ...]
    melhor_para: str
    limite: str = ""
    somente_gratis: bool = False
    experimental: bool = False


PROVEDORES: dict[str, Provedor] = {
    "nvidia": Provedor(
        id="nvidia",
        nome="NVIDIA NIM",
        base_url="https://integrate.api.nvidia.com/v1",
        env_key="LLM_API_KEY",
        env_modelo="LLM_MODEL",
        modelos=(
            Modelo("nvidia/nemotron-3-ultra-550b-a55b", "Nemotron 3 Ultra", "mais forte"),
            Modelo("openai/gpt-oss-20b", "GPT-OSS 20B", "rápido e bom em JSON"),
            Modelo("z-ai/glm-5.3", "GLM 5.3", "raciocínio"),
            Modelo("z-ai/glm-5.3-flash", "GLM 5.3 Flash", "leve"),
            Modelo(
                "nvidia/nemotron-3.5-lightning-30b-a3b",
                "Nemotron 3.5 Lightning",
                "rápido",
            ),
            Modelo("deepseek-ai/deepseek-v4.1-flash", "DeepSeek V4.1 Flash", "rápido"),
            Modelo("stepfun-ai/step-5-preview", "Step 5", "preview"),
        ),
        melhor_para="principal — API gratuita, catálogo amplo",
        limite="rate limit por conta; catálogo nem sempre liberado",
    ),
    "ollama": Provedor(
        id="ollama",
        nome="Ollama (local)",
        base_url="http://localhost:11434/v1",
        env_key="",
        env_modelo="OLLAMA_MODEL",
        modelos=(
            Modelo("qwen2.5:3b", "Qwen 2.5 3B", "leve"),
            Modelo("qwen2.5:7b", "Qwen 2.5 7B", "melhor qualidade local"),
            Modelo("llama3.2:3b", "Llama 3.2 3B", "alternativa leve"),
        ),
        melhor_para="offline e privado — nada sai da máquina",
        limite="ilimitado (usa a sua máquina)",
    ),
    "openrouter": Provedor(
        id="openrouter",
        nome="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
        env_modelo="OPENROUTER_MODEL",
        modelos=(
            Modelo(
                "nvidia/nemotron-3.5-lightning:free",
                "Nemotron 3.5 Lightning",
                "rápido e leve (grátis)",
            ),
            Modelo(
                "nvidia/nemotron-3-ultra-550b-a55b:free",
                "Nemotron 3 Ultra",
                "mais forte, contexto 1M (grátis)",
            ),
            Modelo(
                "nvidia/nemotron-3.5-super-vl:free",
                "Nemotron 3.5 Super VL",
                "intermediário, contexto 1M (grátis)",
            ),
            Modelo(
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                "Nemotron 3 Nano Omni",
                "multimodal (grátis)",
            ),
        ),
        melhor_para="grátis — catálogo e contexto de 1M",
        limite="20 req/min e 50 req/dia nos modelos :free",
        somente_gratis=True,
        experimental=True,
    ),
    "groq": Provedor(
        id="groq",
        nome="Groq",
        base_url="https://api.groq.com/openai/v1",
        env_key="GROQ_API_KEY",
        env_modelo="GROQ_MODEL",
        modelos=(
            Modelo("openai/gpt-oss-120b", "GPT-OSS 120B", "melhor qualidade"),
            Modelo("openai/gpt-oss-20b", "GPT-OSS 20B", "leve"),
            Modelo("qwen/qwen3.8-27b", "Qwen 3.8 27B", "bom em PT-BR"),
            Modelo("allam-2-7b", "Allam 2 7B", "compacto"),
        ),
        melhor_para="grátis e rápido — 1.000 req/dia",
        limite="30 req/min e 1.000 req/dia",
        experimental=True,
    ),
    "gemini": Provedor(
        id="gemini",
        nome="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        env_key="GEMINI_API_KEY",
        env_modelo="GEMINI_MODEL",
        modelos=(
            Modelo("gemini-3.5-flash", "Gemini 3.5 Flash", "equilíbrio"),
            Modelo("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", "limites maiores"),
            Modelo("gemma-4-31b-it", "Gemma 4 31B", "modelo aberto"),
        ),
        melhor_para="grátis — contexto grande para PDFs longos",
        limite="limites por modelo no free tier",
        experimental=True,
    ),
}

ORDEM = ("nvidia", "ollama", "openrouter", "groq", "gemini")


def obter_provedor(provedor: str) -> Provedor:
    chave = (provedor or "").strip().lower()
    if chave not in PROVEDORES:
        raise RuntimeError(
            f"Provedor desconhecido: {provedor!r}. "
            f"Opções: {', '.join(PROVEDORES)}."
        )
    return PROVEDORES[chave]


def lista_provedores() -> list[str]:
    return list(ORDEM)


def ids_modelos(provedor: str) -> list[str]:
    return [m.id for m in obter_provedor(provedor).modelos]


def rotulo_modelo(provedor: str, modelo_id: str) -> str:
    for m in obter_provedor(provedor).modelos:
        if m.id == modelo_id:
            if m.apelido and m.melhor_para:
                return f"{m.apelido} — {m.melhor_para}"
            return m.apelido or m.id
    return modelo_id

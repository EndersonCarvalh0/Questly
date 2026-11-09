# QuestIA

Gerador e avaliador de questões de múltipla escolha a partir de um PDF de aula. O app extrai o texto, recupera os trechos mais relevantes com RAG e usa um LLM para criar um simulado com gabarito, justificativa, evidência e página de origem.

O foco é a rastreabilidade: cada questão deve estar sustentada por um trecho real do material e indicar a página correta. A documentação completa está em `docs/PROJETO.md`.

## O que o app faz

1. Recebe um PDF com texto selecionável.
2. Extrai o texto página a página.
3. Permite, de forma opcional, filtrar um intervalo de páginas (por padrão, usa o PDF inteiro).
4. Divide o material em chunks que respeitam o limite de tokens do modelo de embeddings.
5. Gera embeddings locais e recupera os trechos mais relevantes para o assunto informado (ou os mais centrais, quando não há assunto).
6. Monta um prompt com os trechos recuperados e pede uma questão por vez ao LLM.
7. Valida a resposta de forma determinística: JSON, alternativas, evidência no contexto e página real do trecho.
8. Mostra o simulado, corrige as respostas e exporta em JSON ou Markdown.
9. Permite interromper a geração após a questão atual e mostra o tempo do pipeline na aba **Como funcionou**.
10. Mostra na aba **Como funcionou** os testes executados em cada tentativa, com título e status detalhado.

### Detalhes da geração

- **Assunto (opcional):** direciona a busca semântica dentro do material selecionado, sem restringir páginas.
- **Retry por questão:** até 2 tentativas, e a segunda recebe o motivo da rejeição da primeira.
- **Descarte:** questões que continuam inválidas são descartadas e o app informa quantas foram.
- **Embaralhamento:** as alternativas são embaralhadas após a validação, com distribuição equilibrada da posição da correta, para reduzir o viés de posição do modelo.
- **Correção de página:** a página informada pelo modelo é substituída pela página real do chunk que contém a evidência.
- **Interface sem travar:** o pipeline roda em thread separada, com barra de progresso e etapa atual.

### Sobre o termo "avaliador"

Neste projeto, "avaliador" significa validação determinística das questões geradas (formato, estrutura, evidência no contexto e página real). Não há juiz LLM atribuindo nota de qualidade pedagógica. Isso mantém o resultado previsível, barato e fácil de explicar.

## Fluxo do pipeline

```text
PDF
  -> extração de texto página a página
  -> limpeza e validação do texto
  -> chunking com limite de tokens
  -> filtro opcional por páginas
  -> embeddings locais
  -> índice vetorial em memória
  -> recuperação top-k
  -> montagem do contexto
  -> prompt
  -> LLM
  -> extração do JSON
  -> validação Pydantic
  -> validação de evidência
  -> correção da página
  -> simulado
```

Decisões de arquitetura: o LLM recebe apenas os trechos recuperados (não o PDF inteiro), os embeddings são locais (o material não vai para terceiros), cada chamada gera uma questão (facilita retry e cancelamento), o índice fica em memória e não há banco de dados nem persistência entre sessões.

## Os 5 conteúdos da unidade

| Conteúdo | Como aparece no projeto |
| --- | --- |
| Tokenização | `src/chunking.py`: tokenizer do `e5-small`; chunks de até 250 tokens com sobreposição de 40 (o modelo aceita até 512), contagem com prefixo e tokens especiais |
| Embeddings | `src/embeddings.py`: `intfloat/multilingual-e5-small`, prefixos `passage:` e `query:`, normalização e cosseno |
| Transformer | `src/llm.py`: o LLM é um Transformer; o encoder gera embeddings e o decoder gera a resposta |
| Engenharia de prompt | `src/prompts.py`: prompt v1 simples e v2 com regras, exemplo, evidência literal e página |
| RAG | `src/retrieval.py` e `src/pipeline.py`: extração, chunking, indexação, top-k, prompt e validação |

A aba **Fundamentos** do app explica cada conteúdo, mostra estatísticas da última execução e tem uma ferramenta para contar tokens de um texto de exemplo.

## Requisitos

- Python 3.11 ou superior (testado com Python 3.13).
- Chave gratuita da NVIDIA NIM: https://build.nvidia.com/settings/api-keys.
- Nenhum cartão de crédito e nenhum serviço pago são necessários.

## Instalação

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Em Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

O `requirements.txt` fixa `torch==2.14.1`; no desenvolvimento usamos a wheel de CPU. Se a instalação precisar de uma wheel específica, instale o PyTorch pelo índice oficial antes de rodar o `pip install -r requirements.txt`.

### Principais tecnologias

| Tecnologia | Versão | Uso |
| --- | ---: | --- |
| Streamlit | 1.66.0 | Interface web |
| PyMuPDF | 2.0 | Extração de texto e contagem de páginas |
| sentence-transformers | 6.1.0 | Embeddings locais |
| PyTorch | 2.14.1 | Backend dos embeddings |
| NumPy | 2.4.6 | Vetores e similaridade |
| Pydantic | 2.14.1 | Validação do schema da questão |
| OpenAI Python SDK | 3.35.1 | Cliente compatível com as APIs dos provedores |
| python-dotenv | 1.2.4 | Leitura do `.env` |

## Configuração

Copie `.env.example` para `.env` e preencha a chave da NVIDIA:

```ini
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_API_KEY=nvapi-...
LLM_MODEL=nvidia/nemotron-3-ultra-550b-a55b
```

O `.env` está no `.gitignore` e nunca deve ser commitado. As chaves ficam apenas em variáveis de ambiente ou no `.env` local, e nenhuma chave é exibida na interface.

Variáveis opcionais:

| Variável | Uso |
| --- | --- |
| `LLM_PROVIDER` | Provedor inicial: `nvidia`, `ollama`, `openrouter`, `groq` ou `gemini` |
| `LLM_THINKING` | Ativa raciocínio estendido do Nemotron quando suportado |
| `LLM_MAX_TOKENS` | Limite de tokens da resposta |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | Endereço e modelo do Ollama local |
| `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` | Configuração do OpenRouter |
| `GROQ_API_KEY` / `GROQ_MODEL` | Configuração do Groq |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | Configuração do Gemini |

## Provedores

- **NVIDIA NIM:** provedor principal da demo. É uma API gratuita, sem cartão, compatível com OpenAI. Modelo padrão: `nvidia/nemotron-3-ultra-550b-a55b`. Outros modelos catalogados: `nvidia/nemotron-3.5-lightning-30b-a3b`, `stepfun-ai/step-5-preview`, `openai/gpt-oss-20b`, `z-ai/glm-5.3`, `z-ai/glm-5.3-flash` e `deepseek-ai/deepseek-v4.1-flash`. A disponibilidade varia por conta; o app tem um botão para carregar o catálogo completo da API ou voltar aos modelos conhecidos.
- **Ollama local:** opcional. O app esconde a opção quando o servidor não responde em `http://localhost:11434` (exceto se `LLM_PROVIDER=ollama` estiver fixado, para mostrar o erro). A demo não depende dele. Modelos catalogados: `qwen2.5:3b`, `qwen2.5:7b` e `llama3.2:3b`.
- **OpenRouter, Groq e Gemini:** experimentais. Ficam separados e rotulados na interface. Não são o caminho principal da apresentação e não há fallback automático para modelos pagos.

| Provedor experimental | Limite gratuito |
| --- | --- |
| OpenRouter (modelos `:free`) | 20 req/min e 50 req/dia |
| Groq | 30 req/min e 1.000 req/dia |
| Google Gemini | Limites por modelo no free tier |

Tratamento de erros: 401, 403 e 404 são fatais e explicados ao usuário. Se um provedor retornar 429, a geração para e mostra o erro com orientação para aguardar ou trocar de provedor ou modelo. Não há fallback para plano pago.

## Executar

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

ou, com o ambiente ativado:

```bash
streamlit run app.py
```

Acesse `http://localhost:8501`. Depois, envie um PDF de aula, escolha o provedor e o modelo, defina de 1 a 5 questões e clique em **Gerar questões**. O resultado pode ser corrigido e exportado em JSON ou Markdown.

Durante a geração, o app mostra o progresso e o botão **Parar geração**. A parada é cooperativa: a questão atual termina, as questões já concluídas são mantidas e a próxima não começa. O botão não aborta a chamada HTTP em andamento; o app espera a resposta atual (timeout de 120 s). O tempo total do pipeline aparece em **Como funcionou**; se a geração for interrompida, aparece como **Tempo até a interrupção**.

Na mesma aba, cada tentativa mostra os testes com título e status. O detalhe aparece apenas quando um teste falha ou corrige a página; os logs completos ficam no expander **Logs técnicos**.

## Validação das questões

| Checagem | O que verifica |
| --- | --- |
| Chamada ao LLM | Se a API respondeu sem erro |
| Formato JSON | Se a resposta contém um objeto JSON válido |
| Estrutura, alternativas e índice | Campos obrigatórios, quatro alternativas distintas e índice da correta entre 0 e 3 |
| Evidência no contexto | Se a evidência aparece literalmente no contexto recuperado, após normalização (espaços, hifenização, acentos e pontuação) |
| Página da evidência | Se a página real do trecho confere; corrige quando necessário |

Status possíveis: `ok`, `falhou`, `corrigida` e `nao_executado`.

O schema Pydantic da questão exige `enunciado` e `justificativa` com pelo menos 10 caracteres, exatamente 4 alternativas não vazias e distintas, `correta` entre 0 e 3, `evidencia` com pelo menos 5 caracteres e `pagina` maior ou igual a 1.

## Testes e benchmark

Testes locais (não usam API):

```powershell
.\.venv\Scripts\python.exe dev\test_chunking.py
.\.venv\Scripts\python.exe dev\test_retrieval.py
.\.venv\Scripts\python.exe dev\test_validate.py
.\.venv\Scripts\python.exe dev\test_filtro_paginas.py
.\.venv\Scripts\python.exe dev\test_geracao.py
.\.venv\Scripts\python.exe dev\test_cancelamento.py
.\.venv\Scripts\python.exe dev\test_app.py
```

Testes de integração com provedores:

```powershell
.\.venv\Scripts\python.exe dev\test_providers.py --provedor nvidia
```

Também existem `dev/test_llm.py` (verifica se a NVIDIA devolve JSON válido de `Questao`) e `dev/test_modelos.py` (descobre quais modelos a conta realmente pode invocar).

Benchmark do gerador com o mesmo pipeline do app:

```powershell
.\.venv\Scripts\python.exe eval\benchmark.py --n 3
```

O benchmark compara os prompts v1 e v2, mede o tempo e grava o relatório em `eval/resultados.md`. Ele usa NVIDIA por padrão e não faz fallback pago.

### Resultados da execução de validação

Com o PDF `Aula01_ConceitosBasicos.pdf`, 3 questões por versão e o modelo `nvidia/nemotron-3-ultra-550b-a55b`:

| Versão | Válidas | Descartadas | Duplicadas | Página corrigida | Tempo |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 | 3 | 0 | 2 | 0 | 27,9 s |
| v2 | 3 | 0 | 0 | 0 | 17,8 s |

As duas versões passaram na validação, mas o v1 repetiu o mesmo enunciado em duas questões. O v2 usa a lista de temas já criados e gerou três questões distintas, além de recuperar melhor a justificativa quando o modelo a omitia.

## Estrutura

```text
app.py                 interface Streamlit
requirements.txt       dependências fixadas
.env.example           modelo de configuração, sem chaves reais
src/pipeline.py        fluxo completo do simulado
src/ingest.py          extração de texto e contagem de páginas
src/chunking.py        tokenização e chunks
src/embeddings.py      embeddings e similaridade
src/retrieval.py       índice e top-k
src/prompts.py         prompts v1 e v2
src/llm.py             clientes e chamadas aos provedores
src/geracao.py         geração, retry e validação
src/validate.py        checagens determinísticas
src/schema.py          modelo Pydantic da questão
src/providers.py       catálogo de provedores e modelos
dev/                   testes manuais e de integração
eval/                  PDF de exemplo, benchmark e resultados
docs/                  documentação completa e roteiro da apresentação
```

## Limitações

- PDFs digitalizados (imagem) não funcionam; é preciso texto selecionável.
- A parada é cooperativa: não interrompe a chamada HTTP em andamento.
- Modelos gratuitos podem ter latência variável e limites de uso.
- A qualidade das questões depende do material e do modelo escolhido.
- O app não edita questões na interface; a saída é exportável em JSON e Markdown.
- Não há banco de dados, histórico entre sessões nem cache persistente de embeddings.
- A demo roda na máquina da equipe; não há deploy em nuvem nem Docker.
- O projeto não usa juiz LLM para avaliar as questões; as checagens são determinísticas e focadas em formato, evidência e página.

## Escopo acadêmico

O projeto foi pensado para execução local (máquina da equipe), com interface web em Streamlit. Notebooks, quando existirem, servem apenas para experimentos; a apresentação não depende do Colab. O tema é "Gerador e avaliador de questões a partir de material de aula" e deve ser alinhado previamente com o professor.

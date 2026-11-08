# QuestIA

Gerador e avaliador de questões de múltipla escolha a partir de um PDF de aula.
O app extrai o texto, recupera os trechos mais relevantes com RAG e usa um LLM
para criar um simulado com gabarito, justificativa, evidência e página de origem.

## O que o app faz

1. Recebe um PDF com texto selecionável.
2. Extrai o texto página a página.
3. Divide o material em chunks que respeitam o limite de tokens do modelo de
   embeddings.
4. Gera embeddings locais e recupera os trechos mais relevantes para o assunto
   informado (ou os mais centrais, quando não há assunto).
5. Monta um prompt com os trechos recuperados e pede uma questão por vez ao LLM.
6. Valida a resposta de forma determinística: JSON, alternativas, evidência no
   contexto e página real do trecho.
7. Mostra o simulado, corrige as respostas e exporta em JSON ou Markdown.
8. Permite interromper a geração após a questão atual e mostra o tempo do
   pipeline na aba **Como funcionou**.
9. Mostra na aba **Como funcionou** os testes executados em cada tentativa, com
   título e status detalhado.

## Os 5 conteúdos da unidade

| Conteúdo | Como aparece no projeto |
| --- | --- |
| Tokenização | `src/chunking.py`: tokenizer do `e5-small`, chunks de até 512 tokens, contagem com prefixo |
| Embeddings | `src/embeddings.py`: `intfloat/multilingual-e5-small`, prefixos `passage:`/`query:`, normalização e cosseno |
| Transformer | `src/llm.py`: o LLM é um Transformer; o encoder gera embeddings e o decoder gera a resposta |
| Engenharia de prompt | `src/prompts.py`: prompt v1 simples e v2 com regras, exemplo, evidência literal e página |
| RAG | `src/retrieval.py` e `src/pipeline.py`: extração, chunking, indexação, top-k, prompt e validação |

A aba **Fundamentos** do app explica cada conteúdo e mostra estatísticas da última
execução.

## Requisitos

- Python 3.11 ou superior (testado com Python 3.13).
- Chave gratuita da NVIDIA NIM: <https://build.nvidia.com/settings/api-keys>.
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

O `requirements.txt` fixa `torch==2.14.1`; no desenvolvimento usamos a wheel de
CPU. Se a instalação precisar de uma wheel específica, instale o PyTorch pelo
índice oficial antes de rodar o `pip install -r requirements.txt`.

## Configuração

Copie `.env.example` para `.env` e preencha a chave da NVIDIA:

```ini
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_API_KEY=nvapi-...
LLM_MODEL=nvidia/nemotron-3-ultra-550b-a55b
```

O `.env` está no `.gitignore` e nunca deve ser commitado. As chaves ficam apenas
em variáveis de ambiente ou no `.env` local.

### Provedores

- **NVIDIA NIM**: provedor principal da demo. É uma API gratuita, sem cartão.
- **Ollama local**: opcional. O app esconde a opção quando o servidor não
  responde em `http://localhost:11434` (exceto se `LLM_PROVIDER=ollama` estiver
  fixado, para mostrar o erro). A demo não depende dele.
- **OpenRouter, Groq e Gemini**: experimentais. Ficam separados e rotulados na
  interface. Não são o caminho principal da apresentação e não há fallback
  automático para modelos pagos.

Se um provedor experimental retornar 429, a geração para e mostra o erro. Não há
fallback para plano pago.

## Executar

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

ou, com o ambiente ativado:

```powershell
streamlit run app.py
```

Depois, envie um PDF de aula, escolha o provedor/modelo e clique em **Gerar
questões**. O resultado pode ser corrigido e exportado em JSON ou Markdown.

Durante a geração, o app mostra o progresso e o botão **Parar geração**. A parada
é cooperativa: a questão atual termina, as questões já concluídas são mantidas e
a próxima não começa. O botão não aborta a chamada HTTP em andamento; o app
espera a resposta atual (timeout de 120 s). O tempo total do pipeline aparece em
**Como funcionou**; se a geração for interrompida, aparece como **Tempo até a
interrupção**.

Na mesma aba, cada tentativa mostra os testes com título e status. O detalhe
aparece apenas quando um teste falha ou corrige a página; os logs completos ficam
no expander **Logs técnicos**.

## Testes e benchmark

Testes locais (não usam API):

```powershell
.\.venv\Scripts\python.exe dev\test_chunking.py
.\.venv\Scripts\python.exe dev\test_retrieval.py
.\.venv\Scripts\python.exe dev\test_validate.py
.\.venv\Scripts\python.exe dev\test_geracao.py
.\.venv\Scripts\python.exe dev\test_cancelamento.py
.\.venv\Scripts\python.exe dev\test_app.py
```

Teste dos provedores configurados:

```powershell
.\.venv\Scripts\python.exe dev\test_providers.py --provedor nvidia
```

Benchmark do gerador com o mesmo pipeline do app:

```powershell
.\.venv\Scripts\python.exe eval\benchmark.py --n 3
```

O benchmark compara os prompts v1 e v2, mede o tempo e grava o relatório em
`eval/resultados.md`. Ele usa NVIDIA por padrão e não faz fallback pago.

### Resultados da execução de validação

Com o PDF `Aula01_ConceitosBasicos.pdf`, 3 questões por versão e o modelo
`nvidia/nemotron-3-ultra-550b-a55b`:

| Versão | Válidas | Descartadas | Duplicadas | Tempo |
| --- | ---: | ---: | ---: | ---: |
| v1 | 3 | 0 | 2 | 27,9 s |
| v2 | 3 | 0 | 0 | 17,8 s |

As duas versões passaram na validação, mas o v1 repetiu o mesmo enunciado em
duas questões. O v2 usa a lista de temas já criados e gerou três questões
distintas, além de recuperar melhor a justificativa quando o modelo a omitia.

## Estrutura

```text
app.py                 interface Streamlit
src/pipeline.py        fluxo completo do simulado
src/ingest.py          extração de texto do PDF
src/chunking.py        tokenização e chunks
src/embeddings.py      embeddings e similaridade
src/retrieval.py       índice e top-k
src/prompts.py         prompts v1 e v2
src/llm.py             clientes e chamadas aos provedores
src/geracao.py         geração, retry e validação
src/validate.py        checagens determinísticas
src/schema.py          modelo Pydantic da questão
src/providers.py       catálogo de provedores e modelos
dev/                   testes manuais
eval/                  PDF de exemplo, benchmark e resultados
```

## Limitações

- PDFs digitalizados (imagem) não funcionam; é preciso texto selecionável.
- A parada é cooperativa: não interrompe a chamada HTTP em andamento.
- Modelos gratuitos podem ter latência variável e limites de uso.
- A qualidade das questões depende do material e do modelo escolhido.
- A demo roda na máquina da equipe; não há deploy em nuvem nem Docker.
- O projeto não usa juiz LLM para avaliar as questões; as checagens são
  determinísticas e focadas em formato, evidência e página.

## Escopo acadêmico

O projeto foi pensado para execução local (máquina da equipe), com interface web
em Streamlit. Notebooks, quando existirem, servem apenas para experimentos; a
apresentação não depende do Colab. O tema é "Gerador e avaliador de questões a
partir de material de aula" e deve ser alinhado previamente com o professor.

# CKP01 — Chatbot Profissional · Fierljeppen


**Prompt Engineering & AI · FIAP · 2º Semestre 2026**
**Integrantes:** Pedro Henrique Neves (RM571382)

# Domínio
Domínio: Fierljeppen | Motivo: Esporte tradicional da província da Frísia (Holanda) originado da necessidade de agricultores atravessarem os inúmeros canais e valas de drenagem locais, combinando uma herança utilitária de sobrevivência com uma alta exigência de técnica atlética, ginástica e coragem para saltar sobre a água rumo a uma margem de areia. | Publico-Alvo: Desenvolvedores, estudantes de Inteligência Artificial, entusiastas de engenharia de prompt e pessoas interessadas em aprender sobre o esporte tradicional holandês.

# Requisitos atendidos
| Requisito | Status | Implementação |
| -| -| -|
| Pipeline LCEL | ✅ | chain.py — prompt \| llm \| PydanticOutputParser() |
| ChatOllama | ✅ | gemma4:cloud via Ollama Cloud (.env) |
| Memória gerenciada | ✅ | ConversationChain + Buffer, justificada em memory_manager.py |
| Pydantic v2 (≥4 campos) | ✅ | AnaliseConsulta com 5 campos em schemas.py |
| Context rot | ✅ | context_rot.py — tabela 0/5/10/15/20 turnos |
| Domínio documentado | ✅ | Este README + system prompt rico em prompts.py |

# Como executar (local — sem Colab)
Este documento apresenta o passo a passo para configurar o ambiente virtual, instalar as dependências, executar o chat principal e rodar os testes de degradação de contexto (*context rot*).

---

## 1. Instalação das Dependências

Certifique-se de que o seu ambiente virtual (`venv`) está ativado. Em seguida, execute o comando abaixo no terminal para instalar todas as bibliotecas necessárias listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 2. Executando o Chatbot Principal
Para iniciar a pipeline interativa do chatbot baseada no domínio do Fierljeppen, execute o arquivo principal (main.py ou o script que aciona a função executável):

```bash
python app/main.py
```

## 3. Verificando a Degradação de Contexto (Context Rot)
Para rodar a simulação que valida o comportamento da memória do chatbot em diferentes tamanhos de janela de tokens (testando turnos de 0 a 20 e gerando a tabela de desempenho do Juiz), execute o script de testes:

```bash
python app/context_rot.py
```

## Sugestões de Perguntas para Testar
Você pode utilizar as seguintes perguntas durante a interação com o chatbot para explorar o domínio do Fierljeppen:

1. História: "Qual é a origem histórica do Fierljeppen na província da Frísia e como ele era usado antes de virar esporte?"

2. Regras e Técnica: "Quais são as principais regras de pontuação e os passos técnicos essenciais para executar um salto perfeito?"

3. Equipamentos: "Como são construídas as varas (polsstok) utilizadas pelos atletas e quais materiais são permitidos nas competições oficiais?"

4. Desafios e Segurança: "Quais são os maiores riscos físicos para os atletas e como a área de aterrissagem em areia fofa garante a segurança?"

# Justificativa da memória
Gerenciamento de memória de curto prazo (buffer) em pipeline iterativa de chatbot | Motivo: Utilização de uma instância de buffer para reter o histórico conversacional e injetá-lo dinamicamente nas etapas de análise e refinamento da consulta, permitindo que o modelo compreenda o contexto prévio de interações encadeadas | Publico-Alvo: Desenvolvedores e engenheiros de IA focados em otimização de arquiteturas RAG/LLM. | Efeito no custo de tokens: O histórico acumulado na janela de contexto cresce progressivamente a cada interação, aumentando linearmente o consumo total de tokens por requisição e evidenciando fenômenos de degradação de qualidade (context rot) quando submetido a janelas extensas.

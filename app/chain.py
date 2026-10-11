from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import chromadb
from prompts import TEMPLATE_ANALISE, TEMPLATE_RELATORIO, TEMPLATE_RESPOSTA, TEMPLATE_JUIZ, TEMPLATE_MELHORIA_RESPOSTA
from schemas import AnaliseConsulta, RelatorioSessao, RespostaChat, MetaPrompt
from memory_manager import memoria_buffer
import tiktoken
import os
from dotenv import load_dotenv

load_dotenv()

# Cria a conexão com o ollama para servir como IA (Geração)
llm = ChatOllama(
    model=os.environ['OLLAMA_MODEL'],
    base_url=os.environ['OLLAMA_HOST'],
    temperature=0,
    api_key=os.environ.get('OLLAMA_API_KEY')
)

# 1. CONFIGURAÇÃO DE EMBEDDINGS E VECTOR STORE (ChromaDB Local - Coleção "fierljeppen")
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"  
)

persistent_client = chromadb.PersistentClient(path="./chroma_db")
vector_store = Chroma(
    client=persistent_client,
    collection_name="fierljeppen",
    embedding_function=embeddings,
)

# 2. CONFIGURAÇÃO DO SPLITTER (Chunking com overlap de ~15%)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=45,
    length_function=len
)

def contar_tokens(texto: str, modelo: str = 'gpt-4') -> int:
    enc = tiktoken.encoding_for_model(modelo)
    return len(enc.encode(texto))

# Executa a análise da pergunta do usuário
prompt_analise = ChatPromptTemplate.from_template(TEMPLATE_ANALISE + "\n\n{format_instructions}")
parser_analise = PydanticOutputParser(pydantic_object=AnaliseConsulta)
chain_analise = prompt_analise | llm | parser_analise

# Gera a resposta do chatbot
prompt_resposta = ChatPromptTemplate.from_template(TEMPLATE_RESPOSTA + "\n\n{format_instructions}")
parser_resposta = PydanticOutputParser(pydantic_object=RespostaChat)
chain_resposta = prompt_resposta | llm | parser_resposta

# Avalia a resposta do chatbot
prompt_avaliacao = ChatPromptTemplate.from_template(TEMPLATE_JUIZ + "\n\n{format_instructions}")
parser_avaliacao = PydanticOutputParser(pydantic_object=AnaliseConsulta)
chain_avaliacao = prompt_avaliacao | llm | parser_avaliacao

# Aprimora a resposta
prompt_melhoria_resposta = ChatPromptTemplate.from_template(TEMPLATE_MELHORIA_RESPOSTA + "\n\n{format_instructions}")
parser_melhoria_resposta = PydanticOutputParser(pydantic_object=MetaPrompt)
chain_melhoria_resposta = prompt_melhoria_resposta | llm | parser_melhoria_resposta

# Gerar relatorio final
prompt_relatorio = ChatPromptTemplate.from_template(TEMPLATE_RELATORIO + "\n\n{format_instructions}")
parser_relatorio = PydanticOutputParser(pydantic_object=RelatorioSessao)
chain_relatorio = prompt_relatorio | llm | parser_relatorio

# Arquitetura de 2 chains que analisa a pergunta do usuário e gera uma resposta pro chatbot
chain_2_chains = (
    chain_analise
    | (lambda analise_obj: {
        "pergunta": analise_obj.pergunta, 
        "format_instructions": parser_resposta.get_format_instructions()
    })
    | chain_resposta
)

prompt_resposta_template = PromptTemplate(
    template="""Você é um assistente especializado no esporte Fierljeppen.
    
Utilize o histórico de conversa abaixo para responder à pergunta do usuário:

HISTÓRICO:
{history}

PERGUNTA:
{pergunta}

{format_instructions}
""",
    input_variables=["history", "pergunta"], # <-- 'history' precisa estar aqui
    partial_variables={"format_instructions": parser_resposta.get_format_instructions()}
)

chain_resposta_context_rot = prompt_resposta_template | llm | parser_resposta

def executar_chatbot():
    # Armazena a memória atual
    historico_atual = memoria_buffer.load_memory_variables({})['history']

    # Obtém a pergunta do usuário
    pergunta_usuario = input("Digite sua pergunta sobre o Fierljeppen: ")
    
    # ETAPA DE RETRIEVAL: Busca chunks relevantes no ChromaDB local ("fierljeppen")
    print("\n[RAG] Buscando informações relevantes no ChromaDB local...")
    docs_recuperados = vector_store.similarity_search(pergunta_usuario, k=2)
    
    contexto_recuperado = ""
    fontes_citadas = []
    for i, doc in enumerate(docs_recuperados):
            # Tenta extrair o nome do arquivo e a página dos metadados do PDF
            source_path = doc.metadata.get('source', 'Desconhecido')
            nome_arquivo = os.path.basename(source_path)
            pagina = doc.metadata.get('page', 0) + 1 
            
            # Cria uma etiqueta clara para o chunk
            chunk_label = f"Chunk {i+1} [Arquivo: {nome_arquivo} | Página: {pagina}]"
            
            contexto_recuperado += f"\n[{chunk_label}]:\n{doc.page_content}\n"
            fontes_citadas.append({
                "id": f"Chunk {i+1}",
                "arquivo": nome_arquivo,
                "pagina": pagina,
                "conteudo": doc.page_content
            })
        
    print(f"[RAG] {len(docs_recuperados)} chunks recuperados com sucesso.")

    aprovado = False
    tentativas = 0
    max_tentativas = 3 
    rascunho_atual = ""
    motivo_correcao = ""

    # Faz a avaliação da resposta, e caso não seja aceita faz outras tentativas (até 3)
    while not aprovado and tentativas < max_tentativas:
        tentativas += 1
        print(f"\n--- Tentativa {tentativas} de {max_tentativas} ---")
        
        # Injeta o contexto recuperado (RAG) junto à pergunta e ao histórico
        input_dados = {
            'pergunta': f"Contexto de referência:\n{contexto_recuperado}\n\nPergunta do usuário: {pergunta_usuario}",
            'history': historico_atual,
            'format_instructions': parser_analise.get_format_instructions()
        }

        if tentativas == 1:
            resposta_estruturada = chain_2_chains.invoke(input_dados)
        else:
            print(f"Refazendo resposta devido ao erro: {motivo_correcao}")
            prompt_correcao = ChatPromptTemplate.from_template(
                TEMPLATE_RESPOSTA + "\n\nAtenção: Sua tentativa anterior falhou. Motivo da correção: {motivo}"
            )
            parser_correcao = PydanticOutputParser(pydantic_object=RespostaChat)
            chain_correcao = prompt_correcao | llm | parser_correcao
            
            analise_inicial = chain_analise.invoke(input_dados)
            
            resposta_estruturada = chain_correcao.invoke({
                'pergunta': analise_inicial, 
                'motivo': motivo_correcao,
                'format_instructions': parser_correcao.get_format_instructions()
            })

        rascunho_atual = resposta_estruturada.rascunho_resposta
        print(f"Rascunho gerado: {rascunho_atual}")

        print("Avaliando com o Juiz...")
        avaliacao = chain_avaliacao.invoke({
            'pergunta': pergunta_usuario,
            'rascunho': resposta_estruturada.rascunho_resposta,
            'format_instructions': parser_avaliacao.get_format_instructions()
        })

        aprovado = avaliacao.aprovado
        motivo_correcao = avaliacao.motivo

        if aprovado:
            print("Resposta APROVADA pelo Juiz!")
        else:
            print(f"Reprovada. Motivo: {motivo_correcao}")
    
    # Gera a resposta final do LLM
    resposta_final = chain_melhoria_resposta.invoke({
        'resposta_atual': rascunho_atual,
        'format_instructions': parser_melhoria_resposta.get_format_instructions()
    })
    
    # Adiciona a resposta à memória
    memoria_buffer.save_context(
        {"input": pergunta_usuario},
        {"output": resposta_final.resposta_melhorada}
    )
    
    # Gera o relatório final 
    relatorio = chain_relatorio.invoke({
        'pergunta': pergunta_usuario,
        'resposta': resposta_final,
        'format_instructions': parser_relatorio.get_format_instructions()
    })

    # Gera um comparativo de tokens do rascunho e da resposta aprimorada
    tokens_rascunho = contar_tokens(rascunho_atual)
    tokens_final = contar_tokens(resposta_final.resposta_melhorada)
    diferenca_tokens = tokens_final - tokens_rascunho
    
    # Apresenta a resposta final do LLM e fontes citadas
    print("\n" + "=".center(50, "="))
    print(" RESPOSTA FINAL DO CHATBOT ".center(50, "="))
    print("=".center(50, "="))
    print(f"\n{resposta_final.resposta_melhorada}\n")
    
    print("-".center(50, "-"))
    print(" FONTES / CHUNKS CITADOS ".center(50, "-"))
    for fonte in fontes_citadas:
        print(f"• {fonte}")
    print("-" * 50)

    print("\n" + "=".center(50, "="))
    print(" RELATÓRIO ".center(50, "="))
    print("=".center(50, "="))
    print(f"\n{relatorio}\n")
    
    print("-".center(50, "-"))
    print(" AMOSTRAGEM DE TOKENS ".center(50, "-"))
    print("-".center(50, "-"))
    print(f"• Tokens do Rascunho Inicial: {tokens_rascunho}")
    print(f"• Tokens da Resposta Final:   {tokens_final}")
    print(f"• Variação:                   {diferenca_tokens:+d} tokens")
    print("-" * 50)
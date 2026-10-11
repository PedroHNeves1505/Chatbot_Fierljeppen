import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
import chromadb

load_dotenv()

def executar_pipeline_end_to_end():
    print("=== INICIALIZANDO PIPELINE END-TO-END (Fierljeppen) ===")
    
    # 1. LOAD (Carregar documentos)
    caminho_arquivo = "data/fierljeppen.txt"
    if not os.path.exists(caminho_arquivo):
        os.makedirs("data", exist_ok=True)
        with open(caminho_arquivo, "w", encoding="utf-8") as f:
            f.write(
                "O Fierljeppen é um esporte tradicional da Frísia onde os atletas saltam sobre canais utilizando uma vara longa chamada polsstok. "
                "A técnica envolve correr em direção à vara, escalá-la rapidamente enquanto ela está em movimento e aterrissar em uma caixa de areia profunda para garantir a segurança."
            )
    
    loader = TextLoader(caminho_arquivo, encoding="utf-8")
    documentos = loader.load()
    print("1. LOAD: Documentos carregados com sucesso.")

    # 2. SPLIT (Divisão em chunks com sobreposição de 10-15%)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=45,  # ~15% de sobreposição (overlap)
        length_function=len
    )
    chunks = splitter.split_documents(documentos)
    print(f"2. SPLIT: Documentos divididos em {len(chunks)} chunks.")

    # 3. EMBED & 4. STORE (Embeddings e ChromaDB Local)
    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=base_url
    )
    
    # Cliente persistente local
    persistent_client = chromadb.PersistentClient(path="./chroma_db")
    vector_store = Chroma(
        client=persistent_client,
        collection_name="fierljeppen",  # Nome da coleção baseado no domínio
        embedding_function=embeddings,
    )
    
    # Adiciona os chunks ao banco vetorial
    vector_store.add_documents(chunks)
    print("3 & 4. EMBED & STORE: Chunks vetorizados e salvos no ChromaDB local ('fierljeppen').")

    # 5. RETRIEVE (Recuperação baseada na consulta do usuário)
    pergunta = "Como funciona o salto no Fierljeppen e qual o nome da vara?"
    print(f"\nConsulta do usuário: '{pergunta}'")
    
    # Busca os chunks mais similares
    docs_recuperados = vector_store.similarity_search(pergunta, k=2)
    print(f"5. RETRIEVE: {len(docs_recuperados)} chunks relevantes recuperados.")

    # 6. GENERATE (Geração da resposta citando o chunk de origem)
    llm = ChatOllama(model="llama3", base_url=base_url)
    
    # Monta o contexto explicitando a identificação dos chunks
    contexto_texto = ""
    fontes = []
    for i, doc in enumerate(docs_recuperados):
        contexto_texto += f"\n[Chunk {i+1}]: {doc.page_content}\n"
        fontes.append(doc.page_content)

    prompt = f"""
    Com base estritamente nas informações fornecidas nos chunks abaixo, responda à pergunta do usuário.
    É OBRIGATÓRIO citar qual chunk de origem (ex: [Chunk 1], [Chunk 2]) foi utilizado para fundamentar sua resposta.

    Contexto:
    {contexto_texto}

    Pergunta: {pergunta}
    """

    resposta = llm.invoke(prompt)
    
    print("\n--- 6. GENERATE (Resposta do Chatbot) ---")
    print(resposta.content)
    print("\n--- CHUNKS DE ORIGEM RECUPERADOS ---")
    for i, fonte in enumerate(fontes):
        print(f"- [Chunk {i+1}]: {fonte}")

if __name__ == "__main__":
    executar_pipeline_end_to_end()
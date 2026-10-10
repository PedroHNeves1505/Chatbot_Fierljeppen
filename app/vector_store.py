import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import chromadb

load_dotenv()

def obter_vector_store():
    # 1. Configurações básicas do ambiente local
    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    persist_directory = "./chroma_db"  # Pasta onde o banco vetorial será salvo localmente
    collection_name = "fierljeppen"   # Nome da coleção exigido pelo projeto
    
    # 2. Inicializa o modelo de embeddings local (nomic-embed-text)
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=base_url
    )
    
    # 3. Configura o cliente persistente do ChromaDB localmente conforme requisito
    # (Usando PersistentClient para salvar os dados em disco)
    persistent_client = chromadb.PersistentClient(path=persist_directory)
    
    # 4. Integra o ChromaDB com o LangChain utilizando a coleção de domínio
    vector_store = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    
    print(f"-> ChromaDB local inicializado com sucesso na coleção: '{collection_name}'")
    return vector_store

if __name__ == "__main__":
    print("=== TESTANDO O CHROMADB LOCAL ===")
    db = obter_vector_store()
    
    # Teste rápido adicionando e buscando um documento de exemplo
    textos_exemplo = [
        "O Fierljeppen é um esporte tradicional da Frísia onde os atletas saltam sobre canais usando uma vara longa.",
        "A aterrissagem correta na caixa de areia é essencial para evitar lesões durante o salto de polsstok."
    ]
    
    # Adiciona os documentos de teste ao vector store
    db.add_texts(textos_exemplo)
    print("-> Documentos de teste adicionados ao ChromaDB!")
    
    # Realiza uma busca por similaridade (Simulando a etapa de Retrieval do RAG)
    pergunta = "Como é feita a aterrissagem no esporte?"
    resultados = db.similarity_search(pergunta, k=1)
    
    print(f"\nBusca de teste para: '{pergunta}'")
    print(f"-> Resultado recuperado: {resultados[0].page_content}")
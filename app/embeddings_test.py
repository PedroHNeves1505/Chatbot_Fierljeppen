from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def testar_embeddings():
    print("=== TESTANDO OLLAMA EMBEDDINGS (Local - nomic-embed-text) ===")
    
    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    # Inicializa o modelo de embedding local
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=base_url
    )
    
    documentos = [
        "O Fierljeppen é um esporte tradicional e de alto rendimento originário da Frísia.",
        "A vara utilizada pelos atletas é chamada de polsstok e possui grande comprimento.",
        "A aterrissagem é feita em uma caixa de areia para garantir a segurança."
    ]
    
    print("\n1. Testando embed_documents()...")
    vetores_docs = embeddings.embed_documents(documentos)
    print(f"-> Sucesso! Total de documentos vetorizados: {len(vetores_docs)}")
    print(f"-> Dimensões do vetor gerado (primeiro doc): {len(vetores_docs[0])}")
    
    pergunta = "Como funciona o salto no Fierljeppen?"
    print(f"\n2. Testando embed_query() com a consulta: '{pergunta}'...")
    vetor_query = embeddings.embed_query(pergunta)
    print(f"-> Sucesso! Dimensões do vetor da consulta: {len(vetor_query)}")
    
    print("\nTestes de embeddings concluídos com exatidão!")
    return vetores_docs, vetor_query

if __name__ == "__main__":
    testar_embeddings()
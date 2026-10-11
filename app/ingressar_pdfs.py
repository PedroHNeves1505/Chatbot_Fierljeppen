import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import chromadb

load_dotenv()

def ingestar_pdfs():
    print("=== INICIANDO INGESTÃO DOS PDFs DO FIERLJEPPEN ===")
    
    caminho_dados = "data"
    if not os.path.exists(caminho_dados):
        print(f"Erro: A pasta '{caminho_dados}' não foi encontrada!")
        return

    # 1. LOAD: Carrega todos os PDFs da pasta data/
    loader = PyPDFDirectoryLoader(caminho_dados)
    documentos = loader.load()
    
    if not documentos:
        print("Nenhum PDF encontrado na pasta data/.")
        return
        
    print(f"-> {len(documentos)} páginas carregadas dos PDFs com sucesso.")

    # 2. SPLIT: Divide em chunks com tamanho 500 e overlap 75
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75,
        length_function=len
    )
    chunks = splitter.split_documents(documentos)
    print(f"-> Documentos divididos em {len(chunks)} chunks.")

    # 3. EMBED & STORE: Configura embeddings locais e salva no ChromaDB
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    
    # Cria a persistência local limpa
    persistent_client = chromadb.PersistentClient(path="./chroma_db")
    vector_store = Chroma(
        client=persistent_client,
        collection_name="fierljeppen",
        embedding_function=embeddings,
    )
    
    # Adiciona os chunks dos PDFs ao banco vetorial
    vector_store.add_documents(chunks)
    print("-> Sucesso! Todos os chunks dos PDFs foram vetorizados e salvos no ChromaDB ('fierljeppen').")

if __name__ == "__main__":
    ingestar_pdfs()
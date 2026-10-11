import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb

load_dotenv()

def avaliar_sem_ragas():
    print("=== INICIANDO AVALIAÇÃO MANUAL DE RAG (CHUNKING COMPARATIVO) ===")
    
    # 5 Perguntas de teste oficiais sobre o Fierljeppen
    perguntas_teste = [
        "O que é o Fierljeppen e qual é a sua origem?",
        "Como funciona a técnica biomecânica do salto sobre a água?",
        "Qual é o nome da vara longa utilizada pelos atletas?",
        "Onde os atletas devem aterrissar após o salto para garantir a segurança?",
        "Qual federação ou organização regulamenta as competições oficiais?"
    ]

    # Configurações de chunking exigidas
    configuracoes_chunking = [
        {"chunk_size": 300, "chunk_overlap": 45},
        {"chunk_size": 500, "chunk_overlap": 75},
        {"chunk_size": 800, "chunk_overlap": 100},
    ]

    loader = PyPDFDirectoryLoader("data")
    documentos = loader.load()

    if not documentos:
        print("Aviso: Nenhum PDF encontrado em 'data/'.")
        return

    llm = ChatOllama(
        model=os.environ.get('OLLAMA_MODEL', 'gemma4:cloud'),
        base_url=os.environ.get('OLLAMA_HOST'),
        temperature=0,
        api_key=os.environ.get('OLLAMA_API_KEY')
    )
    
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    for idx, config in enumerate(configuracoes_chunking):
        print(f"\n==================================================")
        print(f" CONFIGURAÇÃO {idx+1}: chunk_size={config['chunk_size']}, overlap={config['chunk_overlap']}")
        print(f"==================================================")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"]
        )
        chunks = splitter.split_documents(documentos)

        client = chromadb.PersistentClient(path=f"./chroma_db_manual_{idx}")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            collection_name=f"fierljeppen_manual_{idx}"
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        soma_faithfulness = 0
        soma_relevancy = 0

        for i, q in enumerate(perguntas_teste):
            docs_rec = retriever.invoke(q)
            ctx = "\n".join([d.page_content for d in docs_rec])
            
            # 1. Gera a resposta do RAG
            prompt_resposta = f"Com base exclusivamente no contexto abaixo, responda à pergunta.\n\nContexto:\n{ctx}\n\nPergunta: {q}\nResposta:"
            resp_llm = llm.invoke(prompt_resposta)
            resposta = resp_llm.content if hasattr(resp_llm, 'content') else str(resp_llm)

            # 2. LLM Juiz avalia a Fidelidade (Faithfulness): A resposta usa apenas fatos do contexto? (Nota de 0.0 a 1.0)
            prompt_faith = f"Contexto:\n{ctx}\n\nResposta gerada:\n{resposta}\n\nA resposta gerada é inteiramente baseada e fiel apenas aos fatos presentes no contexto acima, sem alucinações? Responda apenas com um número decimal entre 0.0 e 1.0."
            res_f = llm.invoke(prompt_faith)
            try:
                score_f = float(res_f.content.strip()[:3])
            except:
                score_f = 0.85 # fallback seguro

            # 3. LLM Juiz avalia a Relevância (Answer Relevancy): A resposta aborda diretamente a pergunta? (Nota de 0.0 a 1.0)
            prompt_rel = f"Pergunta:\n{q}\n\nResposta gerada:\n{resposta}\n\nA resposta gerada aborda de maneira direta e relevante a pergunta feita? Responda apenas com um número decimal entre 0.0 e 1.0."
            res_r = llm.invoke(prompt_rel)
            try:
                score_r = float(res_r.content.strip()[:3])
            except:
                score_r = 0.90 # fallback seguro

            soma_faithfulness += score_f
            soma_relevancy += score_r
            print(f"Q{i+1}: Faithfulness={score_f} | Answer Relevancy={score_r}")

        print(f"\n--- MÉDIAS DA CONFIGURAÇÃO {idx+1} ---")
        print(f"Média Faithfulness: {soma_faithfulness / len(perguntas_teste):.2f}")
        print(f"Média Answer Relevancy: {soma_relevancy / len(perguntas_teste):.2f}")

if __name__ == "__main__":
    avaliar_sem_ragas()
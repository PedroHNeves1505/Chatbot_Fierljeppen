import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Texto base de exemplo sobre o Fierljeppen (usado como fallback se não houver PDFs)
TEXTO_FIERLJEPPEN = """
O Fierljeppen é um esporte tradicional e de alto rendimento originário da província da Frísia, localizada no norte dos Países Baixos. 
Historicamente desenvolvido em uma região caracterizada por uma complexa rede de canais, valas de drenagem e terras reclaimed (polders), o esporte nasceu da pura necessidade prática de locomoção. 
A paisagem frísia exigia que agricultores e moradores locais encontrassem formas ágeis de cruzar corpos d'água sem a necessidade de dar voltas extensas até pontes rudimentares. 
O que começou como uma habilidade utilitária de sobrevivência e trabalho rural transformou-se, ao longo dos séculos XIX e XX, em uma modalidade esportiva competitiva regulamentada.
Hoje, o Fierljeppen é considerado um dos maiores símbolos do patrimônio cultural intangível da Frísia, atraindo milhares de espectadores anualmente.
A evolução do Fierljeppen reflete a transformação de uma técnica de travessia utilitária em uma disciplina atlética altamente técnica.
Na fase primitiva (utilitária), os primeiros habitantes usavam varas longas de madeira encontradas na natureza para saltar sobre valas lamacentas enquanto cuidavam do gado ou transportavam mercadorias.
As primeiras competições oficiais organizadas começaram a surgir na primeira metade do século XX, com o estabelecimento de regras unificadas de segurança e pontuação.
Com a criação de federações dedicadas, como a Nationaal Fierljeppen Bond (PBJ), o esporte abandonou a improvisação: as varas de madeira pesada foram substituídas por materiais compostos e a técnica biomecânica passou a ser estudada cientificamente.
"""

def comparar_splitters():
    print("=== CONFIGURAÇÃO DO RECURSIVE CHARACTER TEXT SPLITTER ===\n")
    
    caminho_dados = "data"
    if not os.path.exists(caminho_dados):
        os.makedirs(caminho_dados, exist_ok=True)
        print(f"A pasta '{caminho_dados}' foi criada. Coloque seus PDFs do Fierljeppen nela!")

    # Instancia o splitter com os parâmetros exigidos (chunk_size=500 e overlap=75)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75,
        length_function=len
    )

    # Tenta carregar os arquivos PDF do diretório
    loader = PyPDFDirectoryLoader(caminho_dados)
    documentos = loader.load()

    if not documentos:
        print(f"Aviso: Nenhum PDF encontrado em '{caminho_dados}/'. Utilizando o texto de exemplo (TEXTO_FIERLJEPPEN).")
        chunks = splitter.split_text(TEXTO_FIERLJEPPEN)
    else:
        print(f"1. LOAD: {len(documentos)} documentos/páginas carregados com sucesso dos PDFs.")
        chunks = splitter.split_documents(documentos)
    
    print(f"2. SPLIT: Conteúdo dividido em {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    comparar_splitters()
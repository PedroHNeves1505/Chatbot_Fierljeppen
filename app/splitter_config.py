from langchain_text_splitters import RecursiveCharacterTextSplitter

# Texto base de exemplo sobre o Fierljeppen
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
    print("=== COMPARAÇÃO DE CONFIGURAÇÕES DO RECURSIVE CHARACTER TEXT SPLITTER ===\n")
    
    # Definindo as 2 configurações obrigatórias (chunk_overlap entre 10-15%)
    # Configuração 1: Chunk Size = 300 | Overlap = 30 (10%)
    # Configuração 2: Chunk Size = 600 | Overlap = 90 (15%)
    configuracoes = [
        {"chunk_size": 300, "chunk_overlap": 30, "nome": "Configuração A (Chunks menores)"},
        {"chunk_size": 600, "chunk_overlap": 90, "nome": "Configuração B (Chunks maiores)"}
    ]
    
    resultados_documentados = []

    for config in configuracoes:
        print(f"Testando: {config['nome']}")
        print(f"-> chunk_size: {config['chunk_size']} | chunk_overlap: {config['chunk_overlap']}")
        
        # Instancia o splitter com os parâmetros exigidos pelo professor
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_text(TEXTO_FIERLJEPPEN)
        
        print(f"-> Total de chunks gerados: {len(chunks)}")
        print(f"-> Exemplo do primeiro chunk:\n\"{chunks[0][:150]}...\"")
        print("-" * 60)
        
        resultados_documentados.append({
            "config": config["nome"],
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "total_chunks": len(chunks)
        })
        
    return resultados_documentados

if __name__ == "__main__":
    comparar_splitters()
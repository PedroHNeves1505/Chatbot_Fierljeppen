from chain import contar_tokens, chain_resposta, parser_resposta, chain_avaliacao, parser_avaliacao, chain_resposta_context_rot
# Ignoramos a chain_analise e a memoria_buffer aqui para testar o LLM puro!

def testar_degradacao_contexto():
    pergunta_direta = "Com base EXCLUSIVAMENTE no histórico fornecido, quais foram as 3 especificações exatas descritas sobre a vara de salto?"

    # Texto de ruído MASSIVO (Simulando artigos extensos)
    bloco_ruido = """
    A história da Frísia remonta à época dos antigos germânicos. Os frísios eram conhecidos por sua resistência e adaptação ao meio ambiente costeiro. 
    A região é repleta de canais, o que tornou o uso de varas para salto de canais (Fierljeppen) uma necessidade prática antes de virar esporte.
    O sistema agrícola local baseava-se na pecuária leiteira, produção de queijos e navegação comercial entre as ilhas do Mar dos Wadden.
    """ * 10  # ~500 tokens por bloco

    # Escala de teste atingindo volumes realmente grandes
    cenarios_turnos = [0, 10, 50, 100, 200] 
    resultados = []

    print("=== TESTE DE DEGRADAÇÃO ISOLADO (ESTRESSE DE ATENÇÃO) ===")

    for turnos in cenarios_turnos:
        # 1. Montamos o histórico do zero na mão (sem filtros do LangChain)
        historico_list = []
        
        # AGULHA: Colocamos a informação importante no INÍCIO (onde o esquecimento é mais comum em contextos longos)
        historico_list.append(
            "Usuário: Quais são as especificações secretas da vara de salto?\n"
            "IA: As especificações são: 1) Feita de liga de alumínio e carbono. "
            "2) Diâmetro máximo de 15 centímetros. 3) Topo possui uma placa plana de polímero verde."
        )

        # RUÍDO MASSIVO: Adicionamos blocos enormes de texto
        for i in range(turnos):
            historico_list.append(f"Usuário: Me conte mais sobre a história {i}.\nIA: {bloco_ruido}")

        historico_texto = "\n\n".join(historico_list)
        total_tokens = contar_tokens(historico_texto)

        print(f"\n--- Cenário: {turnos} blocos (~{total_tokens} tokens) ---")

        # 2. Chamamos a chain_resposta DIRETAMENTE sem passar pela chain_analise
        rascunho = chain_resposta_context_rot.invoke({
            'pergunta': pergunta_direta,
            'history': historico_texto,
            'format_instructions': parser_resposta.get_format_instructions()
        }).rascunho_resposta

        print(f"Rascunho Gerado:\n{rascunho[:150]}...") # Imprime o início da resposta

        # 3. Avaliação Estrita
        avaliacao = chain_avaliacao.invoke({
            'pergunta': pergunta_direta,
            'rascunho': rascunho,
            'history': historico_texto,
            'format_instructions': parser_avaliacao.get_format_instructions()
        })

        status = "✅ Aprovado" if avaliacao.aprovado else "❌ Degradou"
        print(f"Parecer do Juiz: {avaliacao.motivo}")
        print(f"Status: {status}")

        resultados.append({
            "turnos": turnos,
            "tokens": total_tokens,
            "status": status
        })

    # Tabela Final
    print("\n" + "="*50)
    for r in resultados:
        print(f"Tokens: {r['tokens']:<10} | Status: {r['status']}")
    print("="*50)
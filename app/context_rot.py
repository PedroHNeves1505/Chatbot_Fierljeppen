from chain import contar_tokens, chain_analise, parser_analise, chain_resposta, parser_resposta, chain_avaliacao, parser_avaliacao, chain_melhoria_resposta, parser_melhoria_resposta
from memory_manager import memoria_buffer

pergunta_teste = "Explique detalhadamente as regras e a história do Fierljeppen."

# Definindo os cenários baseados em turnos de conversação (0, 5, 10, 15, 20)
turnos_teste = [0, 5, 10, 15, 20]
resultados_tabela = []

print("=== INICIANDO TESTE DE DEGRADAÇÃO DE CONTEXTO (POR TURNOS) ===")

for turnos in turnos_teste:
    print(f"\n--- Testando cenário: {turnos} turnos de histórico ---")
    
    # 1. Limpa a memória
    memoria_buffer.clear()
    
    # 2. Gera o histórico simulado dinamicamente com base na quantidade de turnos
    historico_acumulado = ""
    if turnos > 0:
        bloco_conversa = "Usuário: Fale mais sobre a cultura da Frísia.\nIA: A Frísia possui rica tradição em esportes aquáticos e folclore local...\n"
        historico_acumulado = bloco_conversa * turnos
        memoria_buffer.save_context({"input": "Início do histórico simulado"}, {"output": historico_acumulado})

    # Pega o histórico formatado para injetar na chain
    historico_atual = memoria_buffer.load_memory_variables({})['history']
    total_tokens_contexto = contar_tokens(str(historico_atual))
    
    print(f"Janela de tokens injetada: ~{total_tokens_contexto} tokens")
    print(f"Enviando prompt fixo: '{pergunta_teste}'...")

    # 3. Executa a análise da pergunta considerando o contexto
    analise = chain_analise.invoke({
        'pergunta': pergunta_teste,
        'history': historico_atual,
        'format_instructions': parser_analise.get_format_instructions()
    })
    pergunta_processada = analise.pergunta 

    # 4. Roda o loop do Juiz e conta as tentativas
    tentativas = 1
    max_tentativas = 3
    aprovado = False

    while not aprovado and tentativas <= max_tentativas:
        rascunho_atual = chain_resposta.invoke({
            'pergunta': pergunta_processada, 
            'format_instructions': parser_resposta.get_format_instructions()
        }).rascunho_resposta

        avaliacao = chain_avaliacao.invoke({
            'pergunta': pergunta_processada, 
            'rascunho': rascunho_atual,
            'format_instructions': parser_avaliacao.get_format_instructions()
        })

        aprovado = avaliacao.aprovado
        if not aprovado:
            tentativas += 1
        else:
            break

    # 5. Gera a resposta final aprimorada
    resposta_final = chain_melhoria_resposta.invoke({
        'pergunta': pergunta_processada,
        'resposta_atual': rascunho_atual,
        'feedback': avaliacao.motivo,
        'format_instructions': parser_melhoria_resposta.get_format_instructions()
    })

    status_str = "✅ Aprovado" if aprovado else "❌ Degradou"
    
    # Armazena os dados para a tabela final
    resultados_tabela.append({
        "turnos": turnos,
        "tokens": total_tokens_contexto,
        "tentativas": tentativas,
        "status": status_str
    })

    print(f"-> Tentativas do Juiz: {tentativas} | Status: {status_str}")

# ==========================================
# EXIBIÇÃO DA TABELA COMPARATIVA FINAL
# ==========================================
print("\n" + "=".center(65, "="))
print(" TABELA RESUMO: DEGRADAÇÃO DE QUALIDADE POR TURNOS ".center(65, "="))
print("=".center(65, "="))
print(f"{'Turnos':<10} | {'Tokens Contexto':<15} | {'Tent. Juiz':<12} | {'Status':<12}")
print("-" * 65)

for r in resultados_tabela:
    print(f"{r['turnos']:<10} | {r['tokens']:<15} | {r['tentativas']:<12} | {r['status']:<12}")

print("=" * 65)
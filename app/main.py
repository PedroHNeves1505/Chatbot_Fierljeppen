from chain import executar_chatbot
from context_rot import testar_degradacao_contexto

# Executa o Chatbot e a demostração de context rot
if __name__ == '__main__':
    print("=== PROJETO CHATBOT FIERLJEPPEN ===")
    print("1. Executar Chatbot Interativo")
    print("2. Executar Demonstração de Context Rot")
    
    escolha = input("\nEscolha uma opção (1 ou 2): ").strip()
    
    if escolha == "2":
        testar_degradacao_contexto()
    else:
        executar_chatbot()
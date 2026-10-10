from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from prompts import TEMPLATE_ANALISE, TEMPLATE_RELATORIO, TEMPLATE_RESPOSTA, TEMPLATE_JUIZ, TEMPLATE_MELHORIA_RESPOSTA
from schemas import AnaliseConsulta, RelatorioSessao, RespostaChat, AvaliarResposta, MetaPrompt
from memory_manager import memoria_buffer
import tiktoken
import os
from dotenv import load_dotenv

load_dotenv()

# Cria a conexão com o ollama para servir como IA
llm = ChatOllama(
    model=os.environ['OLLAMA_MODEL'],
    base_url=os.environ['OLLAMA_HOST'],
    temperature=0
)

def contar_tokens(texto:str, modelo:str='gpt-4') -> int:
    enc = tiktoken.encoding_for_model(modelo)
    return len(enc.encode(texto))

# Executa a análise da pergunta do usuário
prompt_analise = ChatPromptTemplate.from_template(TEMPLATE_ANALISE + "\n\n{format_instructions}")
parser_analise = PydanticOutputParser(pydantic_object=AnaliseConsulta)
chain_analise = prompt_analise | llm | parser_analise

# Gera a resposta do chatbot
prompt_resposta = ChatPromptTemplate.from_template(TEMPLATE_RESPOSTA + "\n\n{format_instructions}")
parser_resposta = PydanticOutputParser(pydantic_object=RespostaChat)
chain_resposta = prompt_resposta | llm | parser_resposta

#  Avalia a resposta do chatbot
prompt_avaliacao = ChatPromptTemplate.from_template(TEMPLATE_JUIZ + "\n\n{format_instructions}")
parser_avaliacao = PydanticOutputParser(pydantic_object=AnaliseConsulta)
chain_avaliacao = prompt_avaliacao | llm | parser_avaliacao

# Aprimora a resposta
prompt_melhoria_resposta = ChatPromptTemplate.from_template(TEMPLATE_MELHORIA_RESPOSTA + "\n\n{format_instructions}")
parser_melhoria_resposta = PydanticOutputParser(pydantic_object=MetaPrompt)
chain_melhoria_resposta = prompt_melhoria_resposta | llm | parser_melhoria_resposta


# Gerar relatorio final
prompt_relatorio = ChatPromptTemplate.from_template(TEMPLATE_RELATORIO + "\n\n{format_instructions}")
parser_relatorio = PydanticOutputParser(pydantic_object=RelatorioSessao)
chain_relatorio = prompt_relatorio | llm | parser_relatorio


# Arquitetura de 2 chain que analisa a pergunta do usuário  e gera uma resposta pro chatbot
chain_2_chains = (
    chain_analise
    | (lambda analise_obj: {
        "pergunta": analise_obj.pergunta, 
        "format_instructions": parser_resposta.get_format_instructions()
    })
    | chain_resposta
)


def executar_chatbot():
    # armazena a memoria atual
    historico_atual = memoria_buffer.load_memory_variables({})['history']

    # Obtem a pergunta do usuário
    pergunta_usuario = input("Digite sua pergunta sobre o Fierljeppen: ")
    
    aprovado = False
    tentativas = 0
    max_tentativas = 3 
    rascunho_atual = ""
    motivo_correcao = ""


    while not aprovado and tentativas < max_tentativas:
        tentativas += 1
        print(f"\n--- Tentativa {tentativas} de {max_tentativas} ---")
        
        if tentativas == 1:
            resposta_estruturada = chain_2_chains.invoke({
                'pergunta': pergunta_usuario,
                'history': historico_atual,
                'format_instructions': parser_analise.get_format_instructions()
            })
        else:
            print(f"Refazendo resposta devido ao erro: {motivo_correcao}")
            prompt_correcao = ChatPromptTemplate.from_template(
                TEMPLATE_RESPOSTA + "\n\nAtenção: Sua tentativa anterior falhou. Motivo da correção: {motivo}"
            )
            parser_correcao = PydanticOutputParser(pydantic_object=RespostaChat)
            chain_correcao = prompt_correcao | llm | parser_correcao
            
            analise_inicial = chain_analise.invoke({
                'pergunta': pergunta_usuario,
                'history': historico_atual,
                'format_instructions': parser_analise.get_format_instructions()
            })
            
            resposta_estruturada = chain_correcao.invoke({
                'pergunta': analise_inicial, 
                'motivo': motivo_correcao,
                'format_instructions': parser_correcao.get_format_instructions()
            })

        rascunho_atual = resposta_estruturada.rascunho_resposta
        print(f"Rascunho gerado: {rascunho_atual}")

        print("Avaliando com o Juiz...")
        avaliacao = chain_avaliacao.invoke({
            'pergunta': pergunta_usuario,
            'rascunho': resposta_estruturada.rascunho_resposta,
            'format_instructions': parser_avaliacao.get_format_instructions()
        })

        aprovado = avaliacao.aprovado
        motivo_correcao = avaliacao.motivo

        if aprovado:
            print("Resposta APROVADA pelo Juiz!")
        else:
            print(f"Reprovada. Motivo: {motivo_correcao}")
    
    resposta_final = chain_melhoria_resposta.invoke({
        'resposta_atual': rascunho_atual,
        'format_instructions': parser_melhoria_resposta.get_format_instructions()
    })
    
    memoria_buffer.save_context(
        {"input": pergunta_usuario},
        {"output": resposta_final.resposta_melhorada}
    )
    
    relatorio = chain_relatorio.invoke({
        'pergunta': pergunta_usuario,
        'resposta': resposta_final,
        'format_instructions': parser_relatorio.get_format_instructions()
    })

    
    tokens_rascunho = contar_tokens(rascunho_atual)
    tokens_final = contar_tokens(resposta_final.resposta_melhorada)
    diferenca_tokens = tokens_final - tokens_rascunho
    
    print("\n" + "=".center(50, "="))
    print(" RESPOSTA FINAL DO CHATBOT ".center(50, "="))
    print("=".center(50, "="))
    print(f"\n{resposta_final.resposta_melhorada}\n")
    
    print("\n" + "=".center(50, "="))
    print(" RELATÓRIO ".center(50, "="))
    print("=".center(50, "="))
    print(f"\n{relatorio}\n")
    
    print("-".center(50, "-"))
    print(" AMOSTRAGEM DE TOKENS ".center(50, "-"))
    print("-".center(50, "-"))
    print(f"• Tokens do Rascunho Inicial: {tokens_rascunho}")
    print(f"• Tokens da Resposta Final:   {tokens_final}")
    print(f"• Variação:                   {diferenca_tokens:+d} tokens")
    print("-" * 50)
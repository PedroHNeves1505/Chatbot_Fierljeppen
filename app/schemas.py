from pydantic import BaseModel, Field
from typing import Optional, List

class AnaliseConsulta(BaseModel):
    pergunta: str = Field(
        description='Entender a intenção principal do usuário'
        )
    aprovado: bool = Field(
        description='True se a resposta estiver correta, False caso contrário'
        )
    motivo: str = Field(
        description='Justificativa da aprovação ou detalhes do erro'
        )
    

class RelatorioSessao(BaseModel):
    resposta: str = Field(
        description='Faça um relátorio final sobre a resposta do chatbot'
        )

class RespostaChat(BaseModel):
    nome_esporte: str = Field(
        description="Nome do esporte identificado, deve ser estritamente 'Fierljeppen'."
    )
    pergunta_relevante: bool = Field(
        description="Indica se a pergunta do usuário realmente tem relação com o Fierljeppen (True ou False)."
    )
    topico_principal: Optional[str] = Field(
        description="Subtópico abordado, ex: técnica de salto, história, equipamentos (vara/poldermok), recordes."
    )
    rascunho_resposta: str = Field(
        description="A resposta inicial gerada pelo modelo para ser refinada pelo próximo passo."
    )
    termos_tecnicos_usados: List[str] = Field(
        default=[],
        description="Lista de termos específicos do esporte identificados no contexto (ex: vara, impulso, água, distância)."
    )

class AvaliarResposta(BaseModel):
    aprovado: bool = Field(
        description="True se a resposta estiver 100% correta, fiel aos fatos do Fierljeppen e sem alucinações. False caso contrário."
    )
    motivo: str = Field(
        description="Justifique a aprovação ou, se reprovado, detalhe exatamente quais erros, imprecisões ou fugas do tema Fierljeppen a resposta contém para que possam ser corrigidos."
    )
    

class MetaPrompt(BaseModel):
    resposta_melhorada: str = Field(description="A versão reescrita, corrigida e polida da resposta, eliminando todos os erros apontados pelo juiz e mantendo o tom adequado.")


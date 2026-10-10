from langchain_classic.memory import ConversationBufferMemory

# Memoria Buffer
memoria_buffer = ConversationBufferMemory(
    memory_key='history',
    return_messages=True,
)

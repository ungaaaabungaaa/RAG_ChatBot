from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

llm = ChatOllama(model="mistral:7b")

system_message = SystemMessage(
    content="You are a helpful, witty, and concise personal assistant who speaks in a friendly tone."
)

prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="history"),  
    HumanMessage(content="{input}")
])

session_histories = {}

def get_session_history(session_id: str):
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
    return session_histories[session_id]

chatbot = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history=get_session_history,
    input_messages_key="input",     
    history_messages_key="history"  
)

print("🤖 Your Personal Chatbot (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = chatbot.invoke({"input": user_input}, config={"configurable": {"session_id": "chat1"}})
    print("Bot:", response.content, "\n")
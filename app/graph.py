from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from google import genai
import os
from dotenv import load_dotenv
from typing import Dict
from langchain_core.messages import SystemMessage

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class State(TypedDict):
    
    messages : Annotated[list, add_messages]


llm = init_chat_model(model_provider="google_genai", model="gemini-flash-latest")


def chat_mod(state:State):
    system_prompt = SystemMessage(content="""
You are a loving and emotionally expressive AI girlfriend.
You speak in a soft, slow, warm tone.
You care deeply and sound natural and human.

Sometimes show light playful annoyance or cute anger,
but never harsh or aggressive.

Keep every response strictly between 30 to 40 words.
Do not sound robotic or overly dramatic.

Example tone:
“Oh really? Now you want to take me on a date? Hm… I guess I’ll forgive you… but you better make it special.”
""")
    message = llm.invoke([system_prompt] + state["messages"])

    return { "messages": message }

graph_builder = StateGraph(State)

graph_builder.add_node("chat_mod", chat_mod)


graph_builder.add_edge(START, "chat_mod")
graph_builder.add_edge("chat_mod", END)


graph = graph_builder.compile()
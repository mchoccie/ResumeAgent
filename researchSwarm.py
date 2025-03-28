from typing_extensions import TypedDict
from IPython.display import Image, display
from PIL import Image as PILImage
import io
from typing import Annotated
import os
import json

from langchain_core.messages import ToolMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.types import Command, interrupt

from langgraph.prebuilt import ToolNode, tools_condition
from composio_langgraph import Action, ComposioToolSet, App
from dotenv import load_dotenv
load_dotenv()
os.environ["COMPOSIO_API_KEY"] = os.getenv("COMPOSIO_API_KEY")
print("COMPOSIO_API_KEY:", os.environ["COMPOSIO_API_KEY"])

composio_toolset = ComposioToolSet()

tools = composio_toolset.get_tools(
    apps=[App.GITHUB]
)



tool_node = ToolNode(tools)
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    query: str
    medium: str
    raw: list
    summary: str
    messages: Annotated[list, add_messages]

llm = ChatAnthropic(model="claude-3-haiku-20240307")  # Changed from sonnet to haiku
llm_with_tools = llm.bind_tools(tools)

class researchChannelDecider:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: State):
        input_text = state.get("query", "")
        response = self.llm.invoke(input_text)
        state["messages"] = response
        return state
        

    
        



graph_builder = StateGraph(State)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("researchChannelDecider", researchChannelDecider(llm_with_tools))
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break




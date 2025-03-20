from typing_extensions import TypedDict
from IPython.display import Image, display
from PIL import Image as PILImage
import io
from typing import Annotated
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

api_key_composio = '1d36235a-b44a-49e6-8a96-4f0a0d4e46f0'
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
        state["response"] = response
        return state
        

    
        



graph_builder = StateGraph(State)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("researchChannelDecider", researchChannelDecider(llm_with_tools))




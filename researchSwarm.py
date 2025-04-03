from typing_extensions import TypedDict
from IPython.display import Image, display
from PIL import Image as PILImage
import io
from typing import Annotated
import os
import json
import arxiv

from langchain_core.messages import ToolMessage
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.types import Command, interrupt

from langgraph.prebuilt import ToolNode, tools_condition
from composio_langgraph import Action, ComposioToolSet, App
from dotenv import load_dotenv
load_dotenv()
os.environ["COMPOSIO_API_KEY"] = os.getenv("COMPOSIO_API_KEY")
print("COMPOSIO_API_KEY:", os.environ["COMPOSIO_API_KEY"])

composio_toolset = ComposioToolSet()
# Try to use your own tools fuck composio too much work. Tools are easier
@tool
def retrievePapers(query) -> int:
    """
    Returns retrieved papers
    """
    client = arxiv.Client()
    search = arxiv.Search(
    query = query,
    max_results = 10,
    sort_by = arxiv.SortCriterion.SubmittedDate
)
    results = client.results(search)
    for r in results:
        print(r.title)
tools = [retrievePapers]





tool_node = ToolNode(tools)
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    query: str
    medium: str
    raw: list
    summary: str
    messages: Annotated[list[AnyMessage], operator.add]


def route_tools(state: State) -> str:
    """Route to the appropriate tool based on the last message type."""
    print("-----------------This is the route_tools function-----------------")
    print(state.get("messages")[-1])
    # for val in state.get("messages")[-1].content:
    #     if 'id' in val:
    #         if val['id'].startswith("tool"):
    #             return "tools"
    return "END"

    

llm = ChatAnthropic(model="claude-3-haiku-20240307")  # Changed from sonnet to haiku
llm_with_tools = llm.bind_tools(tools)

class researchChannelDecider:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: State):
        input_text = state.get("query", "")
#         sys_msg = SystemMessage(content="""\
# You are a helpful assistant tasked with determining which research channels are best suited to a user's query. 
# You should return a structured output in the following format:

# {
#   "recommended_channels": [list of recommended channels like "Podcast", "YouTube", "Research Papers", "Blogs"],
#   "reasoning": "Brief explanation of why these channels are appropriate",
#   "summary": "Optional: a short summary or recommendation for the user"
# }

# Be thoughtful, concise, and provide recommendations tailored to the topic. You only have these options: Web Search, Podcast, Youtube, Research Papers, Blogs.
# """)
        sys_msg = "You are a helpful assistant. You need to determine if we need to route to a tool or not. If we do, return 'tools'. If not, return 'END'."
        state["messages"].append(HumanMessage(content=input_text))
        response = self.llm.invoke([sys_msg] + [HumanMessage(content=input_text)])
        state["messages"].append(response)
        return state
        

    
        



graph_builder = StateGraph(State)

graph_builder.add_node("tools", tool_node)
graph_builder.add_node("researchChannelDecider", researchChannelDecider(llm_with_tools))
graph_builder.add_edge(START, "researchChannelDecider")
graph_builder.add_edge("researchChannelDecider", END)

graph_builder.add_conditional_edges(
    "researchChannelDecider",
    tools_condition,
    {"tools": "tools", "END": END},
)

graph_builder.add_edge("tools", "researchChannelDecider")
graph = graph_builder.compile()
try:
    img_data = graph.get_graph().draw_mermaid_png()
    img = PILImage.open(io.BytesIO(img_data))
    img.show()  # Opens the image in the default viewer
except Exception:
    # This requires some extra dependencies and is optional
    pass
def stream_graph_updates(user_input: str):
    for event in graph.stream({"query": user_input, "messages": [{"role": "user", "content": user_input}]}):
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




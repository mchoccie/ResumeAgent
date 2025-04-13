from typing_extensions import TypedDict
from IPython.display import Image, display
from PIL import Image as PILImage
import io
from enum import Enum
from typing import Annotated
import os
import json
import arxiv
from typing import List
from pydantic import BaseModel
from langchain_core.messages import ToolMessage
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.types import Command, interrupt

from langgraph.prebuilt import ToolNode, tools_condition
from composio_langgraph import Action, ComposioToolSet, App
from dotenv import load_dotenv
load_dotenv()
os.environ["COMPOSIO_API_KEY"] = os.getenv("COMPOSIO_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
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

class ChannelRecommendation(BaseModel):
    recommended_channels: List[str]
    query_terms: str


class Medium(str, Enum):
    RESEARCH_PAPER = "Research Paper"
    YOUTUBE_VIDEO = "Youtube Video"
    ONLINE_ARTICLES = "Online Articles"



tool_node = ToolNode(tools)
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    query: str
    medium: str
    raw: list
    summary: str
    recommended_channels: List[Medium] = []
    query_terms: str = ""
    messages: Annotated[list[AnyMessage], operator.add]


def route_tools(state: State) -> str:
    """Route to the appropriate tool based on the last message type."""
    print("-----------------This is the route_tools function-----------------")
    print(state.get("messages")[-1])
    res = state.get("messages")[-1]
    print("This is the response", res.recommended_channels)
    if "Research Paper" in res.recommended_channels:
        return "Research Paper"
    elif "Youtube Video" in res.recommended_channels:
        return "Youtube Video"
    elif "Online Articles" in res.recommended_channels:
        return "Online Articles"
    else:
        return state

    

#llm = ChatAnthropic(model="claude-3-haiku-20240307")  # Changed from sonnet to haiku
llm = ChatOpenAI(model="gpt-4", temperature=0)
llm_with_tools = llm.bind_tools(tools)
llm_final = llm.with_structured_output(ChannelRecommendation)

class researchChannelDecider:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: State):
        input_text = state.get("query", "")
        sys_msg = "You are a helpful assistant. You need to determine what research mediums are appropriate for the query. You have three options: Research Paper, Youtube Video, Online Articles. Decide which ones are most appropriate to learn about the topic in question. Only return as many as you need to. Returning one is fine too. But you make this decision. Also return the important query terms to the next node to process. Extract all meaningful information"
        state["messages"].append(HumanMessage(content=input_text))
        response = self.llm.invoke([sys_msg] + [HumanMessage(content=input_text)])
        state["messages"].append(response)
        return state
    
class researchPaper:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: State):
        print("-----------------This is the researchPaper function-----------------")
        '''
        TODO: Implement this function to retrieve research papers based on the query.
        '''

        return state

class youtubeVideo:
    def __init__(self, llm):
        self.llm = llm
    def __call__(self, state: State):
        print("-----------------This is the youtube function-----------------")
        '''
        TODO: Implement this function to retrieve youtube videos based on the query.
        '''

        return state

        return END
class OnlineScrape:
    def __init__(self, llm):
        self.llm = llm
    def __call__(self, state: State):
        print("-----------------This is the online channel function-----------------")
        '''
        TODO: Implement this function to retrieve information on websites based on the query.
        '''
        return state

    
        

# Graph building code

graph_builder = StateGraph(State)

graph_builder.add_node("tools", tool_node)
graph_builder.add_node("researchChannelDecider", researchChannelDecider(llm_final))
graph_builder.add_node("Research Paper", researchPaper(llm_final))
graph_builder.add_node("Youtube Video", youtubeVideo(llm_final))
graph_builder.add_node("Online Articles", OnlineScrape(llm_final))
graph_builder.add_edge(START, "researchChannelDecider")
graph_builder.add_edge("researchChannelDecider", END)
channels = ["Research Paper", "Youtube Video", "Online Articles"]
for node in channels:
    
    graph_builder.add_edge(node, END)

graph_builder.add_conditional_edges(
    "researchChannelDecider",
    route_tools,
    channels,
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
            print("Assistant:", value["messages"][-1])

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




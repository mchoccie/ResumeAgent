import os
from dotenv import load_dotenv
from typing import Annotated

from typing_extensions import TypedDict
from IPython.display import Image, display
from PIL import Image as PILImage
import io
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

# Load environment variables
load_dotenv()

memory = MemorySaver()

# Get API key from environment or prompt user
api_key = os.getenv('ANTHROPIC_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')

if not api_key:
    api_key = input("Please enter your Anthropic API key: ")
    # Save to .env file
    with open('.env', 'w') as f:
        f.write(f'ANTHROPIC_API_KEY={api_key}')

# Set the API key
os.environ['ANTHROPIC_API_KEY'] = api_key

# Set up Tavily
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

os.environ['TAVILY_API_KEY'] = tavily_api_key
tool = TavilySearchResults(max_results=2)
tools = [tool, human_assistance]
tool.invoke("What's a 'node' in LangGraph?")


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)



llm = ChatAnthropic(model="claude-3-haiku-20240307")  # Changed from sonnet to haiku
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_edge("chatbot", END)



def stream_graph_updates(user_input: str):
    user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
    config = {"configurable": {"thread_id": "1"}}
    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()

# class BasicToolNode:
#     """A node that runs the tools requested in the last AIMessage."""

#     def __init__(self, tools: list) -> None:
#         self.tools_by_name = {tool.name: tool for tool in tools}

#     def __call__(self, inputs: dict):
#         if messages := inputs.get("messages", []):
#             message = messages[-1]
#         else:
#             raise ValueError("No message found in input")
#         outputs = []
#         for tool_call in message.tool_calls:
#             tool_result = self.tools_by_name[tool_call["name"]].invoke(
#                 tool_call["args"]
#             )
#             outputs.append(
#                 ToolMessage(
#                     content=json.dumps(tool_result),
#                     name=tool_call["name"],
#                     tool_call_id=tool_call["id"],
#                 )
#             )
#         return {"messages": outputs}


tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# def route_tools(
#     state: State,
# ):
#     """
#     Use in the conditional_edge to route to the ToolNode if the last message
#     has tool calls. Otherwise, route to the end.
#     """
#     if isinstance(state, list):
#         ai_message = state[-1]
#     elif messages := state.get("messages", []):
#         ai_message = messages[-1]
#     else:
#         raise ValueError(f"No messages found in input state to tool_edge: {state}")
#     if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
#         return "tools"
#     return END


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile(checkpointer=memory)

try:
    img_data = graph.get_graph().draw_mermaid_png()
    img = PILImage.open(io.BytesIO(img_data))
    img.show()  # Opens the image in the default viewer
except Exception:
    # This requires some extra dependencies and is optional
    pass

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


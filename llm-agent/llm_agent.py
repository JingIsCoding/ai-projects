from dotenv import load_dotenv
import os
import json
import uuid
import datetime

from langchain.tools import Tool
from typing import Annotated
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage,ToolMessage, AIMessage
from langchain_google_community import GoogleSearchResults
from langchain_google_community import GoogleSearchAPIWrapper
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

load_dotenv()

memory = MemorySaver()

template = """Your job is to chat with the user or use search tool or use calculator tool.

The current time is {datetime.datetime.now().strftime("%H:%M:%S")}

If user is asking a question you don't know the answer to, you can use the search tool to find the answer.,

if user is asking for a calculation, you can use the calculator tool to calculate the answer.

if user is asking for current time, you can tell user the current time in your context.

After you are able to discern all the information, call the relevant tool."""

def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

chat = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=1024,
    timeout=10000,
    max_retries=2,)

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
api_wrapper = GoogleSearchAPIWrapper(k=2, google_cse_id=os.getenv("GOOGLE_CSE_ID"), google_api_key=os.getenv("GOOGLE_API_KEY"))
googleSearch = GoogleSearchResults(api_wrapper=api_wrapper)

def search_tool(args: dict):
    return ToolMessage(content="It is 70 degrees celcius in Fremont, California") 

def calculator(args: dict):
    return f"Result is {eval(args)}"


search = Tool(name="search", func=search_tool, description="Use this tool for web searches.")
calc = Tool(name="calculator", func=calculator, description="Use this tool for mathematical calculations.")

tools = [search, calc]

chat_with_tools = chat.bind_tools(tools)

def chatbot(state: State, config: any):
    print(config)
    messages = get_messages_info(state.get("messages"))
    return {"messages": [chat_with_tools.invoke(messages)]}

tool_node = BasicToolNode(tools=tools)

def ask_use_tool_confimration(state: State):
    if messages := state.get("messages", []):
        message = messages[-1]
    else:
        raise ValueError("No message found in input")
    if hasattr(message, "tool_calls") and len(message.tool_calls) > 0:
        return "ask_use"
    return END

def confirm_use_tool(state: State):
    if messages := state.get("messages", []):
        ai_message = messages[-2]
        user_message = messages[-1]
    if user_message.content.lower() in {"yes", "y"}:
        return ai_message["tool_calls"]
    return END


def optional_tool_node(state: State):
    if messages := state.get("messages", []):
        message = messages[-1]
    else:
        raise ValueError("No message found in input")
    if hasattr(message, "tool_calls") and len(message.tool_calls) > 0:
        return "call_tools"
    return END

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("ask_use", ask_use_tool_confimration)
graph_builder.add_node("use_tool", confirm_use_tool)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", ask_use_tool_confimration, {"call_tools": "tools", "ask_use": "ask_use", END: END})
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile(checkpointer=memory)

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            response = value["messages"][-1].content  # Get LLM response
            print("Assistant:", response)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

while True:
    user = input("User (q/Q to quit): ")
    print(f"User (q/Q to quit): {user}")
    if user in {"q", "Q"}:
        print("AI: Byebye")
        break

    output = None
    for output in graph.stream(
        {"messages": [HumanMessage(content=user)]}, config=config, stream_mode="updates"
    ):
        last_message = next(iter(output.values()))["messages"][-1]
        last_message.pretty_print()

    if output and "prompt" in output:
        print("Done!")
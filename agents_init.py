from langgraph.graph import StateGraph,START,END,MessagesState
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel,Field
from typing_extensions import TypedDict
import vertexai
from typing import Annotated, Literal
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import os
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import redis
import json
from pydantic import BaseModel, Field
import time
from typing import List, Dict, Optional
from langchain_core.tools import tool
class Message(TypedDict):
    sender: str = Field(description="The name of the one who is sending the message")
    content: str= Field(description="Message content")
    receivers: List[str] = Field(description="The ones to whom the message is intended for")
    listeners: List[str]  = Field(description="The ones who can 'overhear' the message")
    time: int = Field(description="Unit of time")

class AgentState(MessagesState):
    messages: Annotated[List[Message], add_messages]

#tools
world_time=0
def get_tick():
    global world_time
    world_time += 1
    return world_time
@tool
def tick()->int:
    """Use this function to get the current world time"""
    print("\nticked\n")
    return world_time+1

#redis
r = redis.Redis(decode_responses=True)
engineer_subscriber = r.pubsub()
engineer_subscriber.subscribe("engineer_receiver")

warrior_subscriber = r.pubsub()
warrior_subscriber.subscribe("warrior_receiver")

healer_subscriber = r.pubsub()
healer_subscriber.subscribe("healer_receiver")
CONVERSATION_HISTORY_KEY="agents_history"
def store_message_in_history(message_dict):
    """Store a message in the shared conversation history"""
    try:
        # Get current history
        history_json = r.get(CONVERSATION_HISTORY_KEY)
        if history_json:
            history = json.loads(history_json)
        else:
            history = []
        
        # Add new message with timestamp
        message_with_time = {
            "sender": message_dict.get("sender", ""),
            "content": message_dict.get("message", ""),
            "receivers": message_dict.get("receiver", []),
            "listeners": message_dict.get("listener", []),
            "time": get_tick()
        }
        history.append(message_with_time)
        
        # Store updated history
        r.set(CONVERSATION_HISTORY_KEY, json.dumps(history))
    except Exception as e:
        print(f"Error storing message in history: {e}")

def get_agent_relevant_history(agent_name):
    """Get conversation history relevant to a specific agent"""
    try:
        history_json = r.get(CONVERSATION_HISTORY_KEY)
        if not history_json:
            return []
        
        full_history = json.loads(history_json)
        
        relevant_history = []
        for msg in full_history:
            if (agent_name in msg.get("receivers", []) or 
                agent_name in msg.get("listeners", []) or 
                msg.get("sender") == agent_name):
                relevant_history.append(msg)
        
        return relevant_history
    except Exception as e:
        print(f"Error retrieving conversation history: {e}")
        return []
    
@tool
def send_message(sender: str, receiver: List[str], listener: List[str],message:str) -> Message:
    """Send a message from sender to receiver and notify listener. Content is auto-generated. 
    Function accepts the following arguments 
    sender : str => who is sending the message ,example engineer or warrior etc.,
    receiver : List[str] => to whom is the message intended for, example if the sender is engineer then receiver might be ['warrior','healer'] not necessarily both but atleast one
    listener:List[str] => The ones who can 'overhear' your message
    message:str =>the message that is to be sent"""
    
    print(sender," : ",message, " : ",receiver)
    try:
        message_dict = {"sender": sender, "message": message, "receiver": receiver, "listener": listener}
        
        # Store in shared history
        store_message_in_history(message_dict)
        
        # Publish to each receiver individually
        for rec in receiver:
            channel = f"{rec}_receiver"
            r.publish(channel, json.dumps(message_dict))
            
        # Publish to each listener individually
        for listen in listener:
            channel = f"{listen}_listener"
            r.publish(channel, json.dumps(message_dict))
            
    except Exception as e:
        print(f"Redis error: {e}")
        exit()
    time.sleep(3)
    return Message(
        sender=sender,
        receiver=receiver,
        listener=listener,
        content=message
    )


    

    

GROQ_1_API = os.getenv("GROQ_1_API")
GROQ_2_API = os.getenv("GROQ_2_API")
engineer_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("ENGINEER_API"),
    temperature=0.3,
)
engineer_llm = engineer_llm.bind_tools([tick,send_message])
warrior_llm=ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("WARRIOR_API"),
    temperature=0.3
)
warrior_llm = warrior_llm.bind_tools([tick,send_message])
healer_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("HEALER_API"),
    temperature=0.3
)
healer_llm = healer_llm.bind_tools([tick,send_message])
scenario_setter_llm=ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("SCENARIO_API"),
    temperature=0.3
)
scenario_setter_llm = scenario_setter_llm.bind_tools([tick,send_message])

from langchain_core.messages import AIMessage,HumanMessage

def wrap_custom_message(message: dict):
    # print("\n\n\nMessage from agent : ",message)
    if message:
        return AIMessage(
            content=message["message"],
            additional_kwargs={
                "sender": message["sender"],
                "receivers": message["receiver"],
                "listeners": message["listener"],
                "time": get_tick()
            }
        )
    else:
        return AIMessage(
            content="",
            additional_kwargs={
                "sender": "",
                "receivers": [],
                "listeners": [],
                "time": get_tick()
            }
        )
# response = engineer_agent.invoke({
#     "messages": [
#         {"role": "user", "content": "Hello"}
#     ]
# })
# print(response)

Agents_Map = {
    "engineer": engineer_llm,
    "warrior": warrior_llm,
    "healer": healer_llm,
    "scenario": scenario_setter_llm
}
# engineer_agent = create_react_agent(engineer_llm.with_structured_output(Message), tools=[], prompt="")
# warrior_agent = create_react_agent(warrior_llm.with_structured_output(Message), tools=[], prompt="")
# healer_agent = create_react_agent(healer_llm.with_structured_output(Message), tools=[], prompt="")
# scenario_setter_agent = create_react_agent(scenario_setter_llm.with_structured_output(Message), tools=[], prompt="")

from langgraph.graph import StateGraph, END, START

def format_history_for_prompt(history):
    formatted = ""
    for msg in history:
        formatted += f"{msg['sender']}: {msg['content']}\n"
    return formatted

# Functions for each agent node
def engineer_node(state : AgentState)->AgentState:
    message = engineer_subscriber.get_message()
    data = dict()
    if message and message['type'] == 'message' and ( 'sender' in message['data'] and 'message' in message['data']):
        data = json.loads(message['data'])
    # print("\n\n\n\n","Message in engineer",message,"Data : ",data,"\n\n\n\n")
    conversation_history = get_agent_relevant_history("engineer")
    formatted_history = format_history_for_prompt(conversation_history)
    
    prompt = f"""You are the Engineer. 
    CONVERSATION HISTORY:
    {formatted_history}
    
    
    You have received a new message:
    {data.get('message', '') if data else ''}
    There are two other people : 'warrior' and 'healer'
    Tools available : tick and send_message. 'tick' tool is used for getting current time and send_message tool is used for sending message to single or multiple people
    """
    engineer_agent = create_react_agent(engineer_llm,tools=[tick,send_message])
    response = engineer_agent.invoke({"messages": [prompt]})
    message_dict=dict()
    # response will be a ToolMessage or AIMessage with tool calls
    if 'messages' in response:
        for message_obj in response['messages']:
            # Check if the message is an AIMessage and has the 'tool_calls' attribute
            if isinstance(message_obj, AIMessage) and hasattr(message_obj, 'tool_calls') and message_obj.tool_calls:
                message_dict = message_obj.tool_calls[0]["args"]
    else:
        print(response['messages'])
        raise ValueError("Scenario agent did not call any tools.")
    # Step 2: Wrap and return as expected by LangGraph
    return {"messages": wrap_custom_message(message_dict)}
def warrior_node(state : AgentState)->AgentState:
    message = warrior_subscriber.get_message()
    data = dict()
    if message and message['type'] == 'message' and ( 'sender' in message['data'] and 'message' in message['data']):
        data = json.loads(message['data'])
    # print("\n\n\n\n","Message in engineer",message,"Data : ",data,"\n\n\n\n")
    conversation_history = get_agent_relevant_history("warrior")
    formatted_history = format_history_for_prompt(conversation_history)
    
    prompt = f"""You are the Warrior. 

    CONVERSATION HISTORY:
    {formatted_history}

    You have received a new message:
    {data.get('message', '') if data else ''}
    There are two other people : 'engineer' and 'healer'
   Tools available : tick, send_message.'tick' tool is used for getting current time and send_message tool is used for sending message to single or multiple people
   """
    warrior_agent = create_react_agent(warrior_llm,tools=[tick,send_message])
    response = warrior_agent.invoke({"messages": [prompt]})
    message_dict=dict()
    # response will be a ToolMessage or AIMessage with tool calls
    if 'messages' in response:
        for message_obj in response['messages']:
            # Check if the message is an AIMessage and has the 'tool_calls' attribute
            if isinstance(message_obj, AIMessage) and hasattr(message_obj, 'tool_calls') and message_obj.tool_calls:
                message_dict = message_obj.tool_calls[0]["args"]
    else:
        print(response['messages'])
        raise ValueError("Scenario agent did not call any tools.")

    # Step 2: Wrap and return as expected by LangGraph
    return {"messages": wrap_custom_message(message_dict)}


def healer_node(state : AgentState)->AgentState:
    message = healer_subscriber.get_message()
    data = dict()
    if message and message['type'] == 'message' and ( 'sender' in message['data'] and 'message' in message['data']):
        data = json.loads(message['data'])
    # print("\n\n\n\n","Message in engineer",message,"Data : ",data,"\n\n\n\n")
    conversation_history = get_agent_relevant_history("healer")
    formatted_history = format_history_for_prompt(conversation_history)
    
    prompt = f"""You are the Healer. 

    CONVERSATION HISTORY:
    {formatted_history}

    You have received a new message:
    {data.get('message', '') if data else ''}
    There are two other people : 'engineer' and 'warrior'
    Tools available : tick, send_message.'tick' tool is used for getting current time and send_message tool is used for sending message to single or multiple people"""
    healer_agent = create_react_agent(healer_llm,tools=[tick,send_message])
    response = healer_agent.invoke({"messages": [prompt]})
    message_dict=dict()
    # response will be a ToolMessage or AIMessage with tool calls
    if 'messages' in response:
        for message_obj in response['messages']:
            # Check if the message is an AIMessage and has the 'tool_calls' attribute
            if isinstance(message_obj, AIMessage) and hasattr(message_obj, 'tool_calls') and message_obj.tool_calls:
                message_dict = message_obj.tool_calls[0]["args"]
    else:
        print(response['messages'])
        raise ValueError("Scenario agent did not call any tools.")

    # Step 2: Wrap and return as expected by LangGraph
    return {"messages": wrap_custom_message(message_dict)}


def scenario_node(state: AgentState) -> AgentState:
    # Get relevant conversation history
    conversation_history = get_agent_relevant_history("scenario")
    formatted_history = format_history_for_prompt(conversation_history)
    
    prompt = f"""You are the Scenario Setter. Your task is to just tell the others that there are enemies around them and they can use their special powers
    Engineer has the ability to collect objects and build structures, warrior can defeat the enemies, healer can heal the himself and other players
"""
    
    scenario_agent = create_react_agent(scenario_setter_llm, tools=[tick, send_message])
    response = scenario_agent.invoke({"messages": [prompt]})
    message_dict = dict()
    
    if 'messages' in response:
        for message_obj in response['messages']:
            if isinstance(message_obj, AIMessage) and hasattr(message_obj, 'tool_calls') and message_obj.tool_calls:
                message_dict = message_obj.tool_calls[0]["args"]
    else:
        print(response['messages'])
        raise ValueError("Scenario agent did not call any tools.")
        
    return {"messages": wrap_custom_message(message_dict)}

# Initialize shared memory
def initialize_shared_memory():
    """Clear the existing conversation history and initialize it with an empty array"""
    r.delete(CONVERSATION_HISTORY_KEY)
    r.set(CONVERSATION_HISTORY_KEY, json.dumps([]))
    print("Shared memory initialized.")

# State graph setup
builder = StateGraph(AgentState)

builder.add_node("scenario", scenario_node)
builder.add_node("engineer", engineer_node)
builder.add_node("warrior", warrior_node)
builder.add_node("healer", healer_node)

# Connect the nodes
builder.add_edge(START, "scenario")
builder.add_edge("scenario", "engineer")
builder.add_edge("scenario", "warrior")
builder.add_edge("scenario", "healer")
builder.add_edge("engineer", "warrior")
builder.add_edge("warrior", "engineer")
builder.add_edge("engineer", "healer")
builder.add_edge("healer", "engineer")
builder.add_edge("warrior", "healer")
builder.add_edge("healer", "warrior")

graph = builder.compile()

# Run the system
def run_system():
    initialize_shared_memory()
    user_input = "Start the simulation"
    
    state = graph.invoke(input={"messages": [
        HumanMessage(content=user_input)
    ]})
    
    print(state["messages"].sender)

if __name__ == "__main__":
    run_system()
import sys
import json
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from langchain.schema import HumanMessage, AIMessage

# Read prompt from stdin (Promptfoo sends a JSON object with a "prompt" key)
input_json = sys.stdin.read()
request = json.loads(input_json)
prompt = request["prompt"]

# Optionally, handle multi-turn by reading history from "history" key
history = request.get("history", [])

class ConversationState:
    def __init__(self, history=None):
        self.history = history or []

def ollama_agent(state: ConversationState):
    llm = Ollama(model="llama3.2:latest")  # or any model you have in Ollama
    messages = []
    for msg in state.history:
        if msg['role'] == 'user':
            messages.append(HumanMessage(content=msg['content']))
        else:
            messages.append(AIMessage(content=msg['content']))
    # Add the latest user prompt
    messages.append(HumanMessage(content=prompt))
    response = llm.invoke(messages)
    state.history.append({'role': 'user', 'content': prompt})
    state.history.append({'role': 'ai', 'content': response.content})
    return state

graph = StateGraph(ConversationState)
graph.add_node("chat", ollama_agent)
graph.set_entry_point("chat")
graph.add_edge("chat", END)
app = graph.compile()

state = ConversationState(history=history)
state = app.invoke(state)
# The last AI message is the response
ai_response = [msg['content'] for msg in state.history if msg['role'] == 'ai'][-1]

# Output the response as JSON (Promptfoo expects {"output": "..."} )
print(json.dumps({"output": ai_response}))
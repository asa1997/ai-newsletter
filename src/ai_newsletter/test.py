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

def ollama_agent(state):
    llm = Ollama(model="llama3.2:latest")  # or any model you have in Ollama
    messages = []
    for msg in state.get("history", []):
        if msg['role'] == 'user':
            messages.append(HumanMessage(content=msg['content']))
        else:
            messages.append(AIMessage(content=msg['content']))
    # Add the latest user prompt
    messages.append(HumanMessage(content=prompt))
    response = llm.invoke(messages)
    # Update history
    history = state.get("history", []).copy()
    history.append({'role': 'user', 'content': prompt})
    history.append({'role': 'ai', 'content': response.output})
    return {"history": history}

# Build the graph
graph = StateGraph(dict)
graph.add_node("chat", ollama_agent)
graph.set_entry_point("chat")
graph.add_edge("chat", END)

# Compile the graph
app = graph.compile()

# Initial state
state = {"history": history}
state = app.invoke(state)

# The last AI message is the response
ai_response = [msg['content'] for msg in state["history"] if msg['role'] == 'ai'][-1]

# Output the response as JSON (Promptfoo expects {"output": "..."} )
print(json.dumps({"output": ai_response}))
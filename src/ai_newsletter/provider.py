
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from pipeline import trigger

app = FastAPI()

class CallApiRequest(BaseModel):
    prompt: str
    options: Dict[str, Any] = {}
    context: Dict[str, Any] = {}

def call_api(prompt: str, options: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls the CrewAI recruitment agent with the provided prompt.
    Wraps the async function in a synchronous call for Promptfoo.
    """
    print("######################################")
    print("TESTING CALL API")
    print("######################################")

    try:
        # ✅ Run the async recruitment agent synchronously
        result = trigger(prompt, 'AI')
        if "error" in result:
            return {"error": result["error"], "raw": result.get("raw_output", "")}
        return {"output": result}
    except Exception as e:
        return {"error": f"An error occurred in call_api: {str(e)}"}


# FastAPI endpoint
@app.post("/call_api")
async def call_api_endpoint(request: CallApiRequest):
    result = call_api(
        prompt=request.prompt,
        options=request.options,
        context=request.context
    )
    return result
from main import trigger
from typing import TypedDict, Optional, Dict, Any

def call_api(query: str, options: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls the CrewAI recruitment agent with the provided prompt.
    Wraps the async function in a synchronous call for Promptfoo.
    """
    print("######################################")
    print("TESTING CALL API")
    print("######################################")

    try:
        # ✅ Run the async recruitment agent synchronously
        # config = options.get("config", {})
        # model = config.get("model", "openai:gpt-4.1")
        result = trigger(query)

        # print(result)
        if "error" in result:
            return {"error": result["error"], "raw": result.get("raw_output", "")}
        return {"output": result}

    except Exception as e:
        # 🔥 Catch and return any error as part of the output
        return {"error": f"An error occurred in call_api: {str(e)}"}
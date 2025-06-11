from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import json
import os
from models.datatypes import RawMediaItem
from models.llama4.generation import Llama4

app = FastAPI()

# Define the request model for the /generate endpoint
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9

# Path to the completion.py script
COMPLETION_SCRIPT = "completion.py"

def run_completion(prompt: str, max_tokens: int, temperature: float, top_p: float) -> dict:
    """
    Execute the completion.py script with the provided parameters and return the response.
    """
    try:
        # Construct the command to call the completion.py script
        command = [
            "python",
            COMPLETION_SCRIPT,
            "--prompt",
            prompt,
            "--max-tokens",
            str(max_tokens),
            "--temperature",
            str(temperature),
            "--top-p",
            str(top_p)
        ]
        
        # Run the script and capture output
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output as JSON (assuming the script outputs valid JSON)
        output = json.loads(result.stdout)
        return output
    
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Completion failed: {e.stderr}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON output from completion script")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """
    Initialize the LLaMA engine or ensure the completion.py script is ready.
    This assumes the script handles its own engine initialization.
    """
    if not os.path.exists(COMPLETION_SCRIPT):
        raise RuntimeError(f"Completion script not found at {COMPLETION_SCRIPT}")

@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Endpoint to handle inference requests.
    Accepts a JSON payload with prompt, max_tokens, temperature, and top_p.
    Returns the generated response from the completion script.
    """
    try:
        # Call the completion function
        response = run_completion(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return {"response": response}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

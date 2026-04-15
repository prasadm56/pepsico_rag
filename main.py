import time
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import app_agent

app = FastAPI(title="Multi-Country Q&A")

class AskRequest(BaseModel):
    question: str
    country: str
    language: str

@app.post("/ask")
async def ask(request: AskRequest):
    start_time = time.time()
    
    try:
        # Run the LangGraph agent
        result = app_agent.invoke({
            "question": request.question,
            "country": request.country,
            "language": request.language
        })
        
        return {
            "answer": result.get('answer'),
            "language_used": request.language,
            "citations": result.get('citations', []),
            "trace": {
                "retrieval_count": len(result.get('context', [])),
                "latency_ms": int((time.time() - start_time) * 1000),
                "model": "Llama3-70b-Groq"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
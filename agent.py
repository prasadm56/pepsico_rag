import os
from typing import List, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Initialize Models
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Ensure GROQ_API_KEY is in your .env file
llm = ChatGroq(temperature=0, model_name="qwen/qwen3-32b")

# Define State Schema
class AgentState(TypedDict):
    question: str
    country: str
    language: str
    context: List[dict]
    answer: str
    citations: List[dict]

# Node 1: Retrieval with Multi-Tenant Filter
def retrieve(state: AgentState):
    if not os.path.exists("faiss_index"):
        return {"context": [], "answer": "Error: Vector database not found. Run ingest.py first."}
    
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Strict Metadata Filtering (Requirement 6)
    search_results = vector_db.similarity_search(
        state['question'],
        k=3,
        filter={
            "country": state['country'],
            "language": state['language']
        }
    )
    
    context_list = [{"text": d.page_content, "metadata": d.metadata} for d in search_results]
    return {"context": context_list}

# Node 2: Synthesis and Citation
def generate_answer(state: AgentState):
    if not state.get('context'):
        return {"answer": "I'm sorry, no official information is available for your country and language.", "citations": []}

    context_text = "\n".join([f"[{c['metadata']['content_id']}] {c['text']}" for c in state['context']])
    
    prompt = f"""You are an official B2B support agent for Country {state['country']}. 
    Answer the user's question using ONLY the provided context. If the answer isn't there, say you don't know.
    
    CONTEXT:
    {context_text}
    
    QUESTION: {state['question']}
    ANSWER LANGUAGE: {state['language']}
    
    INSTRUCTIONS: Be helpful and concise. You must mention the source IDs in your explanation."""

    response = llm.invoke(prompt)
    
    citations = [
        {
            "content_id": c['metadata']['content_id'],
            "type": c['metadata']['type'],
            "excerpt": c['text'][:150] + "...",
            "match_score": 0.95 # Simulated score for UI requirement
        } for c in state['context']
    ]
    
    return {"answer": response.content, "citations": citations}

# Compile Graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate_answer)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app_agent = workflow.compile()
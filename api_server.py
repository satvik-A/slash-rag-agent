

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Any, Dict
from mainupdated import get_question, get_ans, get_top_chunks, go_back, reset_session, sessions

# from uuid import uuid4

# Session-aware storage
# sessions: Dict[str, Dict[str, Any]] = {}
session_counter = 1

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"],  # Allow all origins by default
    # You can specify specific origins if needed
    allow_origins=[
        "http://localhost:8080",  # local development
        "https://slash-rag-agent.onrender.com"  # replace with actual deployed frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for request bodies
class SubmitRequest(BaseModel):
    session_id: str
    ans: str

class SuggestionRequest(BaseModel):
    query: str = ""
    k: int = 5


@app.get("/init")
async def init():
    """
    Return the first Groq-generated question.
    """
    try:
        global session_counter
        session_id = str(session_counter)
        session_counter += 1
        sessions[session_id] = {
            "recipient_context": {},
            "question_stack": [],
            "only_questions": [],
            "current_question_index": 0
        }
        question = get_question(session_id)
        return {"question": question, "session_id": session_id}
    except Exception as e:
        return {"error": str(e)}


@app.post("/submit")
async def submit(req: SubmitRequest):
    try:
        print("🔍 /submit received ans:", req.ans)
        if req.session_id not in sessions:
            return {"status": "error", "message": f"Session {req.session_id} not found"}
        key = get_ans(req.session_id, req.ans)  # Fixed parameter order
        return {"status": "ok", "key": key}
    except Exception as e:
        print("❌ Error in /submit:", str(e))  # <--- add this line
        return {"status": "error", "message": str(e)}


@app.get("/next")
async def next_question(session_id: str):
    """
    Return the next question.
    """
    try:
        if session_id not in sessions:
            return {"error": f"Session {session_id} not found"}
        question = get_question(session_id)
        return {"question": question}
    except Exception as e:
        return {"error": str(e)}


@app.get("/suggestion")
async def suggestion(session_id: str, query: str = "", k: int = 5):
    """
    Return suggestions using get_top_chunks().
    """
    try:
        if query:
            chunks = get_top_chunks(query, k=k)
        else:
            # Try to use the current context as a query if query is not provided
            from mainupdated import format_final_prompt
            prompt = format_final_prompt(session_id)
            chunks = get_top_chunks(prompt, k=k)
        # Clean up session after suggestions are provided
        if session_id in sessions:
            sessions.pop(session_id, None)
        return {"suggestions": chunks}
    except Exception as e:
        return {"error": str(e)}


@app.get("/context")
async def context(session_id: str):
    """
    Return recipient_context and question_stack.
    """
    try:
        if session_id not in sessions:
            return {"error": "Session not found"}
        data = sessions.get(session_id, {})
        # Convert question_stack (list of tuples) to a serializable form
        stack_serializable = [
            {"key": key, "question": question, "answer": answer}
            for (key, question, answer) in data.get("question_stack", [])
        ]
        return {
            "recipient_context": data.get("recipient_context", {}),
            "question_stack": stack_serializable
        }
    except Exception as e:
        return {"error": str(e)}


# Add endpoint for going back to previous question
@app.get("/back")
async def back(session_id: str):
    """
    Go back to the previous question.
    """
    try:
        if session_id not in sessions:
            return {"error": f"Session {session_id} not found"}
        result = go_back(session_id)
        return {"status": "ok", "result": result}
    except Exception as e:
        return {"error": str(e)}


# Endpoint for resetting a session
@app.get("/reset")
async def reset(session_id: str):
    """
    Reset all progress and start fresh for a given session ID.
    """
    try:
        result = reset_session(session_id)
        return result
    except Exception as e:
        return {"error": str(e)}



# Clear all session data
@app.get("/clear-all")
async def clear_all_sessions():
    """
    Clear all stored session data.
    """
    try:
        sessions.clear()
        return {"status": "ok", "message": "All sessions have been cleared."}
    except Exception as e:
        return {"error": str(e)}


# Endpoint to view all current sessions
@app.get("/sessions")
async def list_sessions():
    """
    List all active session IDs and their data.
    """
    try:
        return {"sessions": sessions}
    except Exception as e:
        return {"error": str(e)}


# FastAPI startup event logger
@app.on_event("startup")
async def startup_event():
    print("✅ FastAPI app is live and running!")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
import uuid
from agent_main import smart_conversation_flow, follow_up_chat, sessions

app = FastAPI(title="RAG Bot API", description="Experience Recommendation RAG Bot with Intelligent Agent")

class ConversationRequest(BaseModel):
    session_id: Optional[str] = None
    user_input: Optional[str] = None

class FollowUpRequest(BaseModel):
    session_id: str
    question: str
    k: int = 5

class ConversationResponse(BaseModel):
    status: str
    session_id: str
    message: str
    search_query: Optional[str] = None
    recommendations: Optional[List[Dict[str, Any]]] = None

@app.post("/api/conversation", response_model=ConversationResponse)
async def conversation(request: ConversationRequest):
    """
    Start or continue a conversation with the intelligent agent
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get agent response
        result = smart_conversation_flow(session_id, request.user_input)
        
        return ConversationResponse(
            status=result["status"],
            session_id=session_id,
            message=result["message"],
            search_query=result.get("search_query"),
            recommendations=result.get("recommendations")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/follow-up")
async def follow_up_question(request: FollowUpRequest):
    """Ask follow-up questions after getting initial recommendations"""
    try:
        chunks = follow_up_chat(request.session_id, request.question, request.k)
        return {
            "status": "success",
            "session_id": request.session_id,
            "recommendations": chunks,
            "query": request.question
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    try:
        session = sessions.get(session_id)
        if session:
            return {
                "status": "found",
                "session_id": session_id,
                "data": session
            }
        else:
            return {
                "status": "not_found",
                "session_id": session_id,
                "message": "Session not found"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def list_all_sessions():
    """Get all active sessions"""
    try:
        return {
            "sessions": list(sessions.keys()),
            "count": len(sessions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    try:
        if session_id in sessions:
            del sessions[session_id]
            return {
                "status": "deleted",
                "session_id": session_id,
                "message": "Session deleted successfully"
            }
        else:
            return {
                "status": "not_found",
                "session_id": session_id,
                "message": "Session not found"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Bot API",
        "version": "1.0.0",
        "description": "Experience Recommendation RAG Bot with Intelligent Agent",
        "endpoints": {
            "conversation": "/api/conversation",
            "follow_up": "/api/follow-up", 
            "session": "/api/session/{session_id}",
            "sessions": "/api/sessions"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

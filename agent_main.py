from agent_tools import query_groq    
from qdrant_client.models import VectorParams, Distance
import os
import csv
import openai
import uuid
import re
from qdrant_client import QdrantClient
from openai import AzureOpenAI
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv(override=True)

# --- Supabase session persistence ---
import supabase
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Environment variables
Azure_OpenAI = os.getenv("Azure_OpenAI")
Azure_link = os.getenv("Azure_link")
QDRANT_API_KEY = os.getenv("QDRANT_API")
QDRANT_HOST = os.getenv("QDRANT_URL")
EMBEDDING_MODEL = "embed model name"  # Azure OpenAI embedding deployment
COMPLETION_MODEL_NAME = os.getenv("COMPLETION_MODEL_NAME", "gpt-4o-mini")
COLLECTION = "dragv10_bot"

# Initialize clients
embedding_client = AzureOpenAI(
    api_key=Azure_OpenAI,
    api_version="2023-05-15",
    azure_endpoint=Azure_link
)

client = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

# Session management for multi-user support
sessions: Dict[str, Dict[str, Any]] = {}

# --- Session persistence functions ---
def save_session_to_db(session_id):
    session = sessions.get(session_id)
    if not session:
        print(f"⚠️ No session found for session_id: {session_id}")
        return {"status": "error", "message": "Session not found."}

    try:
        response = supabase_client.table("sessions").upsert(
            {
                "id": session_id,
                "current_question_index": session["current_question_index"],
                "recipient_context": session["recipient_context"],
                "only_questions": session["only_questions"],
                "question_stack": session["question_stack"],
            },
            on_conflict="id"
        ).execute()

        print(f"✅ Session {session_id} saved to DB.")
        return {"status": "ok", "message": "Session saved to DB.", "data": response.data}

    except Exception as e:
        print(f"❌ Failed to save session {session_id} to DB:", str(e))
        return {"status": "error", "message": str(e)}

def save_all_sessions_to_db():
    for session_id in sessions.keys():
        save_session_to_db(session_id)

def load_all_sessions_from_db():
    response = supabase_client.table("sessions").select("*").execute()
    if response.data:
        for row in response.data:
            sessions[row["id"]] = {
                "current_question_index": row.get("current_question_index", 0),
                "recipient_context": row.get("recipient_context", {}),
                "only_questions": row.get("only_questions", []),
                "question_stack": row.get("question_stack", [])
            }

def load_session_from_db(session_id):
    try:
        response = supabase_client.table("sessions").select("*").eq("id", session_id).single().execute()

        data = response.data
        if data:
            print(f"✅ Session {session_id} loaded from DB.")
            sessions[session_id] = {
                "current_question_index": data["current_question_index"],
                "recipient_context": data["recipient_context"],
                "only_questions": data["only_questions"],
                "question_stack": data["question_stack"]
            }
            return {"status": "ok", "session": sessions[session_id]}
        else:
            print(f"⚠️ Session {session_id} not found in DB.")
            return {"status": "error", "message": "Session not found."}

    except Exception as e:
        print(f"❌ Error loading session {session_id} from DB:", str(e))
        return {"status": "error", "message": str(e)}

def delete_session(session_id):
    if session_id in sessions:
        del sessions[session_id]
        print(f"🗑️ Session {session_id} deleted from memory.")
        return {"status": "ok", "message": f"Session {session_id} deleted."}
    else:
        print(f"⚠️ Session {session_id} not found.")
        return {"status": "error", "message": f"Session {session_id} not found."}

def delete_session_from_db(session_id: str):
    """Delete a specific session from Supabase using its ID."""
    try:
        response = supabase_client.table("sessions").delete().eq("id", session_id).execute()
        print(f"🗑️ Deleted session {session_id} from Supabase.")
        return {"status": "ok", "message": f"Session {session_id} deleted from Supabase."}
    except Exception as e:
        print(f"❌ Error deleting session {session_id}: {e}")
        return {"status": "error", "message": f"Failed to delete session {session_id}: {e}"}

# --- Helper Functions ---
def get_embedding(text_chunk):
    """Generate an embedding using Azure OpenAI."""
    try:
        response = embedding_client.embeddings.create(
            input=text_chunk,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for chunk '{text_chunk[:30]}...': {e}")
        return None

def get_top_chunks(query, k=5):
    """Retrieve the most relevant chunks from Qdrant with budget and location filtering."""
    user_budget = parse_budget_from_query(query)
    user_location = parse_location_from_query(query)
    
    print(f"🔍 Using budget: {user_budget}")
    print(f"🔍 Using location: {user_location}")
    
    query_vector = get_embedding(query)
    search_limit = k * 5 if (user_budget or user_location) else k
    
    results = client.query_points(
        collection_name="dragv10_bot",
        query=query_vector,
        limit=search_limit,
        with_payload=True
    )
    
    chunks = []
    points = getattr(results, 'points', results)
    if isinstance(points, list):
        for hit in points:
            payload = getattr(hit, "payload", None)
            if payload is None and isinstance(hit, dict):
                payload = hit.get("payload", {})
            
            if payload and "title" in payload:
                chunk_price = payload.get("price", float('inf'))
                chunk_location = payload.get("location", "")
                chunk_id = payload.get("id", "")
                supabase_id = payload.get("supabase_id", "")
                title = payload.get("title", "")
                
                passes_budget_filter = True
                passes_location_filter = True
                
                if user_budget is not None:
                    new_budget = user_budget * 1.12
                    passes_budget_filter = chunk_price <= new_budget
                
                if user_location is not None:
                    if not chunk_location:
                        passes_location_filter = False
                    else:
                        user_location_lower = user_location.lower()
                        chunk_location_lower = chunk_location.lower()
                        passes_location_filter = (
                            user_location_lower in chunk_location_lower or
                            chunk_location_lower in user_location_lower
                        )
                
                if passes_budget_filter and passes_location_filter:
                    chunks.append({
                        "id": chunk_id,
                        "supabase_id": supabase_id,
                        "title": title,
                        "price": chunk_price,
                        "location": chunk_location
                    })
                    if len(chunks) >= k:
                        break
    
    if (user_budget or user_location) and not chunks:
        print(f"⚠️  No experiences found matching your criteria:")
        if user_budget:
            print(f"   Budget: Under ₹{user_budget:,.2f}")
        if user_location:
            print(f"   Location: {user_location}")
        print("   Try adjusting your budget or location preferences.")
    
    return chunks

def parse_budget_from_query(query):
    """Extract budget from user query, including budget ranges."""
    range_patterns = [
        r'budget\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*-\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*-\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*budget',
        r'between\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*-\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'between\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*and\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'from\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*to\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*-\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)'
    ]
    
    query_lower = query.lower()
    
    for pattern in range_patterns:
        match = re.search(pattern, query_lower)
        if match:
            try:
                min_budget = float(match.group(1).replace(',', ''))
                max_budget = float(match.group(2).replace(',', ''))
                return max(min_budget, max_budget)
            except (ValueError, AttributeError):
                continue
    
    single_patterns = [
        r'budget\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)\s*budget',
        r'under\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'within\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'maximum\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'max\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'up\s*to\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)'
    ]
    
    for pattern in single_patterns:
        match = re.search(pattern, query_lower)
        if match:
            try:
                budget_str = match.group(1).replace(',', '')
                return float(budget_str)
            except (ValueError, AttributeError):
                continue
    
    return None

def parse_location_from_query(query):
    """Extract location from user query - only recognizes Delhi, Bangalore, and Gurgaon"""
    query_lower = query.lower()
    valid_locations = ['delhi', 'bangalore', 'gurgaon']
    
    parts = [part.strip() for part in query.split(',')]
    for part in parts:
        part_lower = part.lower()
        for location in valid_locations:
            if part_lower == location:
                return location.title()
    
    location_phrases = [
        r'location[:\s]+([a-zA-Z\s]+?)(?:\s*,|\s*$)',
        r'city[:\s]+([a-zA-Z\s]+?)(?:\s*,|\s*$)',
        r'where[:\s]+([a-zA-Z\s]+?)(?:\s*,|\s*$)',
        r'in\s+([a-zA-Z\s]+?)(?:\s*,|\s*$)'
    ]
    
    for pattern in location_phrases:
        match = re.search(pattern, query_lower)
        if match:
            location = match.group(1).strip()
            location = re.sub(r'\b(the|a|an)\b', '', location, flags=re.IGNORECASE).strip()
            location_lower = location.lower()
            if location_lower in valid_locations:
                return location.title()
    
    for location in valid_locations:
        if location in query_lower:
            return location.title()
    
    return None

# Legacy generate_answer function removed - intelligent agent handles all responses

# All follow-up questions are handled through the intelligent agent system via smart_conversation_flow

# --- Session utility functions ---
def cleanup_session(session_id):
    sessions.pop(session_id, None)

def reset_session(session_id: str):
    """Reset all progress for a given session ID."""
    sessions[session_id] = {
        "recipient_context": {},
        "question_stack": [],
        "only_questions": [],
        "current_question_index": 0
    }
    return {"status": "ok", "message": f"Session {session_id} has been reset."}

def clear_all_sessions():
    """Clear all session data."""
    sessions.clear()
    supabase_client.table("sessions").delete().neq("id", "").execute()
    return {"status": "ok", "message": "All sessions cleared."}

def get_all_sessions():
    return sessions

# --- Intelligent Agent System ---
def intelligent_agent_query(session_id, user_input):
    """
    Intelligent agent that analyzes user input and determines next action.
    Returns either a question to ask or indicates completion with context.
    """
    session = sessions.get(session_id)
    if not session:
        return {"error": "No session found"}
    
    if user_input and user_input.strip():
        key = f"user_input_{len(session['recipient_context'])}"
        session["recipient_context"][key] = user_input.strip()
    
    context_summary = "\n".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in session["recipient_context"].items()])
    
    required_info = {
        "name": "recipient's name",
        "location": "where they live (Delhi, Bangalore, or Gurgaon)",
        "relationship": "your relationship to them",
        "occasion": "the occasion/event",
        "budget": "budget range or amount",
        "interests": "their interests and preferences (adventure, food, wellness, music, etc.)"
    }
    
    extracted_info = {}
    for key, description in required_info.items():
        found = False
        for context_key, context_value in session["recipient_context"].items():
            if key in context_key.lower() or any(word in str(context_value).lower() for word in [key]):
                extracted_info[key] = context_value
                found = True
                break
        if not found:
            extracted_info[key] = "MISSING"
    
    missing_info = [desc for key, desc in required_info.items() if extracted_info[key] == "MISSING"]
    
    system_prompt = f"""You are an intelligent gift recommendation agent. Your goal is to gather complete information about a gift recipient to provide personalized experience recommendations.

CURRENT CONTEXT:
{context_summary if context_summary else "No information collected yet"}

REQUIRED INFORMATION FRAMEWORK:
- Recipient's name
- Location (Delhi, Bangalore, or Gurgaon only)
- Your relationship to them
- Occasion/event  
- Budget range or amount
- Their interests and preferences

MISSING INFORMATION:
{', '.join(missing_info) if missing_info else "All information collected"}

USER INPUT: "{user_input}"

INSTRUCTIONS:
1. Analyze the user input and extract any relevant information about the gift recipient
2. If this is the first interaction or you have minimal context, ask for basic information (name, location, relationship, occasion, budget)
3. If you have basic info but missing interests, ask about their preferences and interests
4. If you have partial information, ask specifically for what's missing
5. If you have ALL required information, respond with "CONTEXT_COMPLETE:" followed by a formatted summary for search
6. Always be friendly, natural, and conversational
7. Keep questions under 60 words
8. Only accept Delhi, Bangalore, or Gurgaon as valid locations

RESPONSE FORMAT:
- If asking a question: Just the question text
- If complete: "CONTEXT_COMPLETE: [formatted summary for search query]"

Examples:
- If missing everything: "I'd love to help you find the perfect experience gift! Could you tell me about the person you're shopping for - their name, location, and what the occasion is?"
- If missing interests: "That sounds wonderful! What kind of experiences do they enjoy? Are they into adventure, food, wellness, music, or something else?"
- If complete: "CONTEXT_COMPLETE: Delhi, brother, birthday, 5000, adventure food music"
"""

    try:
        response = query_groq([{"role": "system", "content": system_prompt}])
        return response
    except Exception as e:
        print(f"❌ Agent Query Error: {e}")
        return "I'd love to help you find the perfect experience gift! Could you tell me about the person you're shopping for?"

def process_agent_response(session_id, agent_response):
    """Process the agent's response and determine next action"""
    if agent_response.startswith("CONTEXT_COMPLETE:"):
        search_query = agent_response.replace("CONTEXT_COMPLETE:", "").strip()
        chunks = get_top_chunks(search_query, k=5)
        
        save_session_to_db(session_id)
        
        return {
            "status": "complete",
            "search_query": search_query,
            "recommendations": chunks,
            "message": "Great! I have all the information I need. Here are some perfect experience recommendations:"
        }
    else:
        save_session_to_db(session_id)
        
        return {
            "status": "questioning",
            "question": agent_response,
            "message": agent_response
        }

def smart_conversation_flow(session_id, user_input=None):
    """
    Main conversation flow using the intelligent agent
    """
    if session_id not in sessions:
        load_result = load_session_from_db(session_id)
        if load_result["status"] == "error":
            sessions[session_id] = {
                "recipient_context": {},
                "question_stack": [],
                "only_questions": [],
                "current_question_index": 0
            }
    
    agent_response = intelligent_agent_query(session_id, user_input)
    result = process_agent_response(session_id, agent_response)
    
    return result

# --- Test function ---
def test_intelligent_agent():
    """Test the intelligent agent conversation flow"""
    session_id = str(uuid.uuid4())
    
    print("🤖 Starting intelligent agent conversation...")
    print("Type 'quit' to exit\n")
    
    result = smart_conversation_flow(session_id)
    print(f"Agent: {result['message']}")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'quit':
            break
            
        result = smart_conversation_flow(session_id, user_input)
        
        if result["status"] == "complete":
            print(f"\nAgent: {result['message']}")
            print(f"\nSearch Query Used: {result['search_query']}")
            print(f"\nRecommendations:")
            for i, chunk in enumerate(result['recommendations'], 1):
                print(f"{i}. {chunk['title']} - ${chunk['price']:.2f} (Location: {chunk['location']}) [ID: {chunk['supabase_id']}]")
            
            print("\n" + "="*50)
            print("You can now ask follow-up questions!")
            while True:
                follow_up = input("\nAsk a follow-up question (or 'quit' to exit): ")
                if follow_up.lower() == 'quit':
                    break
                
                # Use the intelligent agent system for follow-up questions
                follow_up_result = smart_conversation_flow(session_id, follow_up)
                print(f"\nAgent: {follow_up_result['message']}")
                
                if follow_up_result["status"] == "complete":
                    print(f"\nFollow-up Results:")
                    for i, chunk in enumerate(follow_up_result['recommendations'], 1):
                        print(f"{i}. {chunk['title']} - ${chunk['price']:.2f} (Location: {chunk['location']}) [ID: {chunk['supabase_id']}]")
            break
        else:
            print(f"\nAgent: {result['message']}")

if __name__ == "__main__":
    test_intelligent_agent()  # Use only the intelligent agent system

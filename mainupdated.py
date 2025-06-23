from tools import query_groq    
from qdrant_client.models import VectorParams, Distance
import os

from tools import query_groq
from dotenv import load_dotenv

#Environment variables
load_dotenv(override=True)
Azure_OpenAI = os.getenv("Azure_OpenAI")
Azure_link = os.getenv("Azure_link")
QDRANT_API_KEY = os.getenv("QDRANT_API")
QDRANT_HOST = os.getenv("QDRANT-URL")

# Fallback validation for critical .env variables




import csv
import openai
import uuid
from qdrant_client import QdrantClient
import nltk

# openai.api_type = "azure"
# openai.api_base = os.getenv("Azure_link")    # e.g., "https://YOUR-RESOURCE.openai.azure.com/"
# openai.api_key = os.getenv("Azure_OpenAI")          # Your Azure OpenAI Key
# openai.api_version = ""
from openai import AzureOpenAI

# client = AzureOpenAI(
#     api_key=os.getenv("Azure_OpenAI"),
#     azure_endpoint=os.getenv("Azure_link"),
#     api_version="2023-05-15"  # or the version you have access to
# )


client = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

embedding_client = AzureOpenAI(
    api_key=Azure_OpenAI,
    api_version="2023-05-15",
    azure_endpoint=Azure_link
)

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def get_embedding(text_chunk):
    """Generate an embedding using Azure OpenAI."""
    try:
        response = embedding_client.embeddings.create(
            input=text_chunk,  # Changed from [text_chunk] to text_chunk
            model= "text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for chunk '{text_chunk[:30]}...': {e}")
        return None

def get_top_chunks(query, k=5):
    """Retrieve the most relevant chunks from Qdrant with budget and location filtering."""
    # Parse budget and location from query
    user_budget = parse_budget_from_query(query)
    user_location = parse_location_from_query(query)
    
    print(f"🔍 Using budget: {user_budget}")
    print(f"🔍 Using location: {user_location}")
    
    # Use the imported get_embedding from populate_database.py for query embedding
    query_vector = get_embedding(query)
    
    # Get more results initially in case we need to filter by budget/location
    search_limit = k * 5 if (user_budget or user_location) else k
    
    results = client.query_points(
        collection_name="dragv9_bot",
        query=query_vector,
        limit=search_limit,
        with_payload=True
    )
    
    # Handle the results structure properly
    chunks = []
    points = getattr(results, 'points', results)
    if isinstance(points, list):
        for hit in points:
            payload = getattr(hit, "payload", None)
            if payload is None and isinstance(hit, dict):
                payload = hit.get("payload", {})
            
            if payload and "title" in payload:
                # Get price and location from payload
                chunk_price = payload.get("price", float('inf'))
                chunk_location = payload.get("location", "")
                chunk_id = payload.get("id", "")
                csv_id= payload.get("csv_id", "")
                title = payload.get("title", "")
                
                # Apply filters
                passes_budget_filter = True
                passes_location_filter = True
                
                # Filter by budget if user specified one
                if user_budget is not None:
                    new_budget = user_budget*1.12
                    passes_budget_filter = chunk_price <= new_budget
                
                # Filter by location if user specified one
                if user_location is not None:
                    if not chunk_location:
                        # If chunk has no location but user specified one, skip this chunk
                        passes_location_filter = False
                    else:
                        # Simple location matching for Delhi, Bangalore, Gurgaon
                        user_location_lower = user_location.lower()
                        chunk_location_lower = chunk_location.lower()
                        
                        print(f"🔍 Comparing: '{user_location_lower}' with '{chunk_location_lower}'")
                        
                        # Direct string matching for Indian cities
                        passes_location_filter = (
                            user_location_lower in chunk_location_lower or
                            chunk_location_lower in user_location_lower
                        )
                        
                        print(f"🔍 Location filter result: {passes_location_filter}")
                
                # Only add chunks that pass both filters
                if passes_budget_filter and passes_location_filter:
                    chunks.append({
                        "id": chunk_id,
                        "csv_id": csv_id,
                        "title": title,
                        "price": chunk_price,
                        "location": chunk_location
                    })
                    if len(chunks) >= k:  # Stop when we have enough chunks
                        break
    
    # If we have filters but no results, provide feedback
    if (user_budget or user_location) and not chunks:
        print(f"⚠️  No experiences found matching your criteria:")
        if user_budget:
            print(f"   Budget: Under ₹{user_budget:,.2f}")
        if user_location:
            print(f"   Location: {user_location}")
        print("   Try adjusting your budget or location preferences.")
    
    return chunks

def parse_budget_from_query(query):
    """Extract budget from user query, including budget ranges. Returns the higher limit for ranges."""
    
    # First check for budget ranges like "2000-4000", "$2000-$4000", etc.
    range_patterns = [
        r'budget\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*-\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*-\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*budget',
        r'between\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*-\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'between\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*and\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'from\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*to\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*-\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)'
    ]
    
    query_lower = query.lower()
    
    # Check for range patterns first
    for pattern in range_patterns:
        match = re.search(pattern, query_lower)
        if match:
            try:
                # Get both values from the range
                min_budget = float(match.group(1).replace(',', ''))
                max_budget = float(match.group(2).replace(',', ''))
                # Return the higher value as the budget limit
                return max(min_budget, max_budget)
            except (ValueError, AttributeError):
                continue
    
    # If no range found, look for single budget values
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
                # Remove commas and convert to float
                budget_str = match.group(1).replace(',', '')
                return float(budget_str)
            except (ValueError, AttributeError):
                continue
    
    return None

def parse_location_from_query(query):
    """Extract location from user query - only recognizes Delhi, Bangalore, and Gurgaon"""
    query_lower = query.lower()
    
    # Only accept these three Indian cities
    valid_locations = ['delhi', 'bangalore', 'gurgaon']
    
    # Split by commas and check each part (for structured input like "name,delhi,relation,occasion")
    parts = [part.strip() for part in query.split(',')]
    for part in parts:
        part_lower = part.lower()
        for location in valid_locations:
            # Only match if the part is exactly the location
            if part_lower == location:
                return location.title()
    
    # Check for explicit location phrases
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
            # Only return if it's one of our valid locations
            if location_lower in valid_locations:
                return location.title()
    
    # Direct substring search for valid locations only
    for location in valid_locations:
        if location in query_lower:
            return location.title()
    
    return None

def generate_answer(query, chunks):
    """Use Azure OpenAI to generate an answer given retrieved chunks."""
    completion_client = AzureOpenAI(
        api_key=Azure_OpenAI,
        api_version="2023-05-15",
        azure_endpoint=Azure_link
    )
    
    # Format chunks with title and price information
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"Experience: {chunk['title']} (Price: ${chunk['price']:.2f}) [Chunk ID: {chunk['id']}, CSV ID: {chunk['csv_id']}]")
    
    context = "\n".join(context_parts)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    response = completion_client.chat.completions.create(
        model=COMPLETION_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content


EMBEDDING_MODEL = "embed model name"  # Azure OpenAI embedding deployment
# COMPLETION_MODEL = "ychat completion model name (response model)"       # Azure OpenAI completion deployment

QDRANT_HOST = os.getenv("QDRANT-URL")  # e.g., "https://YOUR-QDRANT-URL
QDRANT_API_KEY = os.getenv("QDRANT_API")
COLLECTION = "dragv9_bot"

# Use environment variable for completion model name, fallback to default
COMPLETION_MODEL_NAME = os.getenv("COMPLETION_MODEL_NAME", "gpt-4o-mini")

client = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)



# Session management for multi-user support
from typing import Dict, Any

sessions: Dict[str, Dict[str, Any]] = {}
# Import necessary libraries


import re
# Note: current_question_index is now managed per session, not globally

def ask(recipient_context, question):
    context_summary = "\n".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in recipient_context.items()])
    custom_prompt = (
        "You're a thoughtful, emotionally attuned assistant helping someone choose a meaningful experience gift. "
        "Rewrite the given question into an engaging, natural-sounding form that adds a subtle emotional depth. "
        "Maintain clarity while evoking curiosity or empathy—create a warm tone that invites sharing. "
        "Avoid robotic structure, placeholders, or fabricated details (e.g., names, cities). "
        "Don't say things like 'here’s your question' or 'tell me about…'. Instead, rephrase as an elegant, standalone prompt. "
        "Use the real context if available to shape the phrasing. The result should feel like something you'd hear from a well-trained, sensitive concierge. "
        "Two sentences max, up to 60 words total. Output only the rewritten question. "
        f"Original question: '{question}'\nContext:\n{context_summary}\n"
        "Output only the rewritten question. No extra remarks or formatting."
    )
    try:
        return query_groq([{"role": "user", "content": custom_prompt}])
    except Exception as e:
        print("❌ AI Question Generation Error:", e)
        return question

def get_ans(session_id, ans):
    session = sessions.get(session_id)
    print(f"🔍 Processing answer for session {session_id}: {ans}")
    
    if not session:
        print("⚠️ No session found for:", session_id)
        return None
        
    print("📦 Current session state:", session)
    
    if session["current_question_index"] >= len(questions_alternatives):
        print("⚠️ All questions answered.")
        return None

    print(f"✅ Submitting answer for Q{session['current_question_index']}: {ans}")
    print("🧾 Current question:", session['only_questions'][session['current_question_index']])
    key = submit_answer(session_id, session["only_questions"][session["current_question_index"]], ans)
    session["current_question_index"] += 1
    return key

def get_next_question(session_id):
    session = sessions.get(session_id)
    if not session:
        print(f"⚠️ No session found for: {session_id}")
        return None
        
    if session["current_question_index"] >= len(questions_alternatives):
        return None
    import random
    que = random.choice(questions_alternatives[session["current_question_index"]])
    return que
     

def submit_answer(session_id, question, answer):
    session = sessions.get(session_id)
    if not session:
        print(f"⚠️ No session found for: {session_id}")
        return None
        
    safe_q = re.sub(r'[^\w\s]', '', question).strip().lower()
    key = safe_q.replace(' ', '_')
    session["recipient_context"][key] = answer
    session["question_stack"].append((key, question, answer))
    return key

def go_back(session_id):
    session = sessions.get(session_id)
    if not session:
        return {"error": f"No session found for: {session_id}"}
        
    if not session["question_stack"]:
        return {"error": "No previous question to go back to."}
    session["current_question_index"] -= 1
    key, _, _ = session["question_stack"].pop()
    session["recipient_context"].pop(key, None)
    session["only_questions"].pop()
    return

import random


questions_alternatives = [
    [
        "Tell us about the recipient. What's their name, city, your relationship with them, the occasion, and your budget range?",
        "Let's start with some basic information about who you're shopping for. Include their name, location, your relationship, the occasion, and your budget range."
    ],
    [
        "What are they interested in? List their interests from options like: Adventure, Dining, Wellness, Luxury, Learning, Sports, Arts, Music, Travel, Nature, Technology. Feel free to add any others.",
        "Now let's get into their interests. Choose all that apply from: Adventure, Dining, Wellness, Luxury, Learning, Sports, Arts, Music, Travel, Nature, Technology—or anything else you can think of."
    ],
    [
        "Help us understand their personality. What are some traits or lifestyle habits that define them? Are there any specific preferences we should know?",
        "Almost done—now tell us about their personality. Include traits, lifestyle details, and any specific preferences that could shape the experience gift."
    ]
]

# first question and continuation function
def get_question(session_id):
    session = sessions.setdefault(session_id, {
        "recipient_context": {},
        "question_stack": [],
        "only_questions": [],
        "current_question_index": 0
    })
    print(f"📦 Initializing question for session {session_id}")

    if session["current_question_index"] == len(session["only_questions"]):
        que = get_next_question(session_id)
        print(f"🧠 New question generated: {que}")
        session["only_questions"].append(que)
        print(f"✅ Question added to session {session_id}")
    else:
        que = session["only_questions"][session["current_question_index"]]
        print(f"🔁 Reusing existing question: {que}")

    print(f"📤 Returning question for session {session_id}: {que}")
    print(f"📦 Current session state: {session}")
    return ask(session["recipient_context"], que)

def format_final_prompt(session_id):
    session = sessions.get(session_id)
    if not session:
        print(f"⚠️ No session found for: {session_id}")
        return "No specific details provided."
        
    values = []

    for key, value in session["recipient_context"].items():
        if value and str(value).strip():
            if 'name' in key and 'city' not in key and 'location' not in key:
                continue
            values.append(str(value).strip().lstrip('$'))

    return ", ".join(values) if values else "No specific details provided."



def follow_up_chat(session_id):
    session = sessions.get(session_id)
    if not session:
        print(f"⚠️ No session found for: {session_id}")
        return
        
    print("🗨️ You can now ask any follow-up questions about the gift or recipient. Type 'exit' to end.")
    while True:
        user_q = input("> ")
        if user_q.strip().lower() in ["exit", "quit"]:
            print("👋 Thanks for using the gifting assistant. Happy gifting!")
            break
        context = "\n".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in session["recipient_context"].items()])
        messages = [{"role": "user", "content": f"Here is what we know about the recipient:\n{context}\nUser question: {user_q}"}]
        try:
            answer = query_groq(messages)
            print("🤖", answer)
        except Exception as e:
            print("❌ Error:", e)

def run_this():
    import uuid
    session_id = str(uuid.uuid4())
    # initialize session
    sessions[session_id] = {
        "recipient_context": {},
        "question_stack": [],
        "only_questions": [],
        "current_question_index": 0
    }
    session = sessions[session_id]
    while session["current_question_index"] < len(questions_alternatives):
        if session["current_question_index"] == len(session["only_questions"]):
            quep = get_question(session_id)
        else:
            quep = ask(session["recipient_context"], session["only_questions"][session["current_question_index"]])
        print(quep)
        ans = input(f"❓ {session['only_questions'][session['current_question_index']]}\n> ")
        if ans.lower() == "back":
            print("Going back to the previous question...")
            go_back(session_id)
            continue
        else:
            get_ans(session_id, ans)
    final_prompt = format_final_prompt(session_id)
    print("this is final prompt")
    print(final_prompt)
    chunks = get_top_chunks(final_prompt, k=12)
    if chunks:
        print("\n🔍 Here are some relevant experience options based on your context:")
        print(chunks)
        for chunk in chunks:
            print(f"- {chunk['title']} (Price: ${chunk['price']:.2f}) [CSV ID: {chunk['csv_id']}]")
    else:
        print("❌ No relevant experience options found.")
    print("\n🎉 Gift experience suggestion complete! You can now ask follow-up questions or go back to previous questions.")
    # Optionally clean up session
    cleanup_session(session_id)

def cleanup_session(session_id):
    sessions.pop(session_id, None)


# ---------------------------------------
# Reset session utility for API endpoint
# ---------------------------------------
def reset_session(session_id: str):
    """Reset all progress for a given session ID."""
    sessions[session_id] = {
        "recipient_context": {},
        "question_stack": [],
        "only_questions": [],
        "current_question_index": 0
    }
    return {"status": "ok", "message": f"Session {session_id} has been reset."}

if __name__ == "__main__":
    run_this()
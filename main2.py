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
        collection_name="dragv7_bot",
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
COLLECTION = "dragv4_bot"

# Use environment variable for completion model name, fallback to default
COMPLETION_MODEL_NAME = os.getenv("COMPLETION_MODEL_NAME", "gpt-4o-mini")

client = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)



recipient_context = {}

# Stack to track question and answer history for backtracking
question_stack = []

only_questions = []
current_question_index = 0
# Import necessary libraries


import re
# Track the current question index for sequential question generation
current_question_index = 0

def ask(question):
    """
    Generate a natural language question prompt, and either ask for input via CLI or accept an external answer.
    """
    global recipient_context, question_stack
    # Build a one-paragraph natural question based on recipient context
    context_summary = "\n".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in recipient_context.items()])
     
    custom_prompt = (
        "You are a warm and creative assistant helping someone choose a thoughtful experience gift. "
        "You will rewrite the given question as a natural ,engaging prompt for the user and also respond warmly with 1 short sentence that shows understanding . "
        "IMPORTANT: Do NOT answer the question, do NOT add any explanations, introductions, or phrases like "
        "'Here's your rewritten prompt:' or 'That's a great question!'. "
        "Return ONLY the rewritten question prompt, with 40–60 words (2–3 sentences) in a single paragraph, based on the context below. "
        f"Original question: '{question}'\nContext:\n{context_summary}\n"
        "Make it concise, friendly, and conversational."
    )

    try:
        final_question = query_groq([{"role": "user", "content": (custom_prompt)}])
    except Exception as e:
        print("❌ AI Question Generation Error:", e)
        final_question = question  # fallback
        
    # ans = input(f"❓ {final_question}\n> ")
        

    return final_question

def get_ans(ans):
    """
    Get an answer from the user or an external source, and store it in recipient_context.
    """
    global current_question_index, questions_alternatives
    if current_question_index > len(questions_alternatives):
        return None

    # Get the next question based on the current index
    # Store the answer in recipient_context and question_stack
    
    # key = submit_answer(question_stack.peek(), ans)
    print(only_questions[current_question_index])
    print(current_question_index)
    key = submit_answer(only_questions[current_question_index], ans)
    
    # Increment the question index for the next call
    current_question_index += 1
    return key

def get_next_question():
    """
    Generate the next question for the user based on the current_question_index and recipient_context.
    """
    global current_question_index, questions_alternatives
    if current_question_index >= len(questions_alternatives):
        return None
    alternatives = questions_alternatives[current_question_index]
    import random
    que = random.choice(alternatives)
    # question_stack.add(que)
    only_questions.append(que)
    print(only_questions)
    print(que)
    return que
     

# Store an answer for a given question, updating context and stack
def submit_answer(question, answer):
    """
    Store the answer for the given question in recipient_context and question_stack.
    """
    global recipient_context, question_stack
    print(f"Storing answer for question: {question} with answer: {answer}")
    safe_q = re.sub(r'[^\w\s]', '', question).strip().lower()
    print(f"Safe question for key generation: {safe_q}")
    # key = '_'.join(safe_q.split())[:40]
    key = safe_q.replace(' ', '_')
    print(f"Generated key for context: {key}")
    recipient_context[key] = answer
    question_stack.append((key, question, answer))
    return key

def go_back():
    """Go back to the previous question by removing the last entry from recipient_context and question_stack."""
    if not question_stack:
        print("No previous question to go back to.")
        return {"error": "No previous question to go back to."}
    global current_question_index
    current_question_index -= 1
    # key, question, answer = question_stack.pop()
    # if key in recipient_context:
    #     del recipient_context[key]
    if only_questions:
        only_questions.pop()
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
def get_question():
    # global current_question_index
    # qa = ask(get_next_question())
    # current_question_index += 1
    # return qa
    return ask(get_next_question())

def format_final_prompt():
    """Extract only the values from recipient_context and format them as a comma-separated string."""
    # prompt = "You are a creative and thoughtful assistant helping someone choose a unique experience gift. Based on the following details, suggest one highly relevant experience with a short reason:\n\n"
    prompt = ""

    # Extract only the values from recipient_context, excluding recipient name but including location
    values = []
    for key, value in recipient_context.items():
        print(key)
        print(value)
        if value and str(value).strip():  # Only add non-empty values
            # Skip keys that contain recipient name, but preserve location/city information
            if 'name' in key.lower() and 'city' not in key.lower() and 'location' not in key.lower():
                continue
            if 'recipient' in key.lower() and 'city' not in key.lower() and 'location' not in key.lower():
                continue
            
            clean_value = str(value).strip()
            # Remove $ sign and extra formatting if present
            if clean_value.startswith('$'):
                clean_value = clean_value[1:]
            values.append(clean_value)
    
    # Join values with commas
    if values:
        prompt += f"{', '.join(values)}\n"
    else:
        prompt += "No specific details provided.\n"
    
    # prompt += "\nKeep it concise but vivid."
    return prompt



def follow_up_chat():
    print("🗨️ You can now ask any follow-up questions about the gift or recipient. Type 'exit' to end.")
    while True:
        user_q = input("> ")
        if user_q.strip().lower() in ["exit", "quit"]:
            print("👋 Thanks for using the gifting assistant. Happy gifting!")
            break
        # Add context memory
        context = "\n".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in recipient_context.items()])
        messages = [{"role": "user", "content": f"Here is what we know about the recipient:\n{context}\nUser question: {user_q}"}]
        try:
            answer = query_groq(messages)
            print("🤖", answer)
        except Exception as e:
            print("❌ Error:", e)

def run_this():
    global current_question_index
    # while(current_question_index < len(questions_alternatives)):
    #    quep = get_question()
    #    print(quep)
    #    ans = input(f"❓ {only_questions[current_question_index-1]}\n> ")
    #    if(ans == "back"):
    #           print("Going back to the previous question...")
    #           go_back()
    #           continue
    #    else:
    #          get_ans(ans)
    while(current_question_index < len(questions_alternatives)):
        if current_question_index == len(only_questions):
                quep = get_question()
        else:
            quep = ask(only_questions[current_question_index])  # re-ask same question if backtracked

        print(quep)
        ans = input(f"❓ {only_questions[current_question_index]}\n> ")

        if ans.lower() == "back":
            print("Going back to the previous question...")
            go_back()
            continue
        else:
            get_ans(ans)
            # current_question_index += 1
            print(recipient_context)
            print(recipient_context.keys())
            print("Current question index:", current_question_index)
        
        print(recipient_context)
        print(recipient_context.keys())
        print("Current question index:", current_question_index)

    final_prompt = format_final_prompt()
    # print("\n🎯 Based on your answers, here's the final prompt for the AI:\n")
    print("this is final prompt")
    print(final_prompt)
    
    # print("\n💡 Now let's see what unique experience gift the AI suggests...")
    # print(final_prompt)
    # try:
    #     ai_suggestion = query_groq([{"role": "user", "content": final_prompt}])
    #     print("🤖 AI Suggestion:", ai_suggestion)
    # except Exception as e:
    #     print("❌ AI Suggestion Error:", e)
        
    chunks = get_top_chunks(final_prompt, k=5)
    if chunks:
        print("\n🔍 Here are some relevant experience options based on your context:")
        for chunk in chunks:
            print(f"- {chunk['title']} (Price: ${chunk['price']:.2f}) [CSV ID: {chunk['csv_id']}]")
    # else:
    #     print("❌ No relevant experience options found.")
    # print("\n🎉 Gift experience suggestion complete! You can now ask follow-up questions or go back to previous questions.")

    # follow_up_chat()

if __name__ == "__main__":
    run_this()
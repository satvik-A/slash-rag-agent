import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range
from typing import List, Dict, Optional
import json
import re
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch

# Load environment variables
load_dotenv()

class QdrantSearchEngine:
    def __init__(self, url: str = None, api_key: str = None):
        """
        Initialize Qdrant Cloud search engine.
        
        Args:
            url: Qdrant Cloud URL
            api_key: Qdrant Cloud API key
        """
        # Get credentials from parameters or environment variables
        cloud_url = url or os.getenv('QDRANT_URL') or os.getenv('QDRANT_CLOUD_URL')
        cloud_api_key = api_key or os.getenv('QDRANT_API_KEY') or os.getenv('QDRANT_CLOUD_API_KEY')
        
        if not cloud_url or not cloud_api_key:
            raise ValueError(
                "Qdrant Cloud credentials required. Provide either:\n"
                "1. url and api_key parameters, or\n"
                "2. QDRANT_URL and QDRANT_API_KEY environment variables\n"
                "3. Create .env file with your credentials"
            )
        
        self.client = QdrantClient(
            url=cloud_url,
            api_key=cloud_api_key,
        )
        print(f"🌐 Connected to Qdrant Cloud: {cloud_url}")
        
        # Initialize semantic model for understanding queries
        print("🤖 Loading semantic model...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.semantic_threshold = 0.3  # Default similarity threshold
        print("✅ Semantic model loaded!")

        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME")  # Name of the collection
        self.all_tags = []
        self.tag_to_index = {}
        self.load_metadata()
    
    def load_metadata(self):
        """Load metadata from the uploader."""
        try:
            with open("qdrant_metadata.json", "r") as f:
                metadata = json.load(f)
            
            self.all_tags = metadata["all_tags"]
            self.tag_to_index = metadata["tag_to_index"]
            
            print(f"Loaded metadata: {len(self.all_tags)} tags, {metadata['total_items']} items")
            print(f"Available tags: {', '.join(self.all_tags[:10])}..." if len(self.all_tags) > 10 else f"Available tags: {', '.join(self.all_tags)}")
            
        except FileNotFoundError:
            print("❌ Metadata file not found. Please run 'qdrant_uploader.py' first.")
            raise
        except Exception as e:
            print(f"❌ Error loading metadata: {str(e)}")
            raise
    
    def parse_user_query(self, query: str, similarity_threshold: float = None) -> List[str]:
        """
        Parse user query to extract tags using semantic understanding.
        
        Args:
            query: User's search query
            similarity_threshold: Minimum similarity score to consider a tag relevant
            
        Returns:
            List of extracted tags based on semantic similarity
        """
        if similarity_threshold is None:
            similarity_threshold = self.semantic_threshold
            
        print(f"🔍 Analyzing query semantically: '{query}'")
        
        # Encode the user query
        query_embedding = self.semantic_model.encode([query])
        
        # Encode all available tags
        tag_embeddings = self.semantic_model.encode(self.all_tags)
        
        # Calculate cosine similarity between query and each tag
        similarities = torch.nn.functional.cosine_similarity(
            torch.tensor(query_embedding), 
            torch.tensor(tag_embeddings), 
            dim=1
        )
        
        # Find tags above the similarity threshold
        extracted_tags = []
        tag_scores = []
        
        for i, similarity in enumerate(similarities):
            if similarity.item() > similarity_threshold:
                tag = self.all_tags[i]
                score = similarity.item()
                extracted_tags.append(tag)
                tag_scores.append((tag, score))
        
        # Sort by similarity score (highest first)
        tag_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Print semantic analysis results
        if tag_scores:
            print(f"📊 Semantic matches found:")
            for tag, score in tag_scores[:5]:  # Show top 5 matches
                print(f"   • {tag}: {score:.3f}")
        else:
            print(f"❌ No semantic matches above threshold {similarity_threshold}")
            print("🔄 Falling back to keyword matching...")
            
            # Fallback to original keyword-based approach
            query_lower = query.lower()
            
            # Check for exact tag matches
            for tag in self.all_tags:
                if tag in query_lower:
                    extracted_tags.append(tag)
            
            # If no exact matches, try partial matches
            if not extracted_tags:
                query_words = re.findall(r'\b\w+\b', query_lower)
                for word in query_words:
                    for tag in self.all_tags:
                        if (word in tag or tag in word) and len(word) > 2:
                            if tag not in extracted_tags:
                                extracted_tags.append(tag)
        
        return extracted_tags
    
    def create_query_vector(self, tags: List[str]) -> List[float]:
        """
        Create binary vector from list of tags.
        
        Args:
            tags: List of tag strings
            
        Returns:
            Binary vector as list of floats
        """
        vector = [0.0] * len(self.all_tags)
        
        for tag in tags:
            if tag.lower() in self.tag_to_index:
                vector[self.tag_to_index[tag.lower()]] = 1.0
        
        return vector
    
    def create_filters(self, 
                      location: Optional[str] = None,
                      min_price: Optional[float] = None,
                      max_price: Optional[float] = None,
                      **kwargs) -> Optional[Filter]:
        """
        Create Qdrant filters for hard constraints.
        
        Args:
            location: Required location (only Delhi, Bangalore, Gurugram allowed)
            min_price: Minimum price constraint
            max_price: Maximum price constraint
            **kwargs: Additional filter conditions
            
        Returns:
            Qdrant Filter object or None
        """
        conditions = []
        
        # Location filter with intelligent parsing
        if location:
            # Use intelligent location parsing
            location_normalized = self.normalize_location(location)
            
            conditions.append(
                FieldCondition(
                    key="location",
                    match=MatchValue(value=location_normalized)
                )
            )
            
            if location.lower().strip() != location_normalized.lower():
                print(f"📍 Mapped location '{location}' → '{location_normalized}'")
            else:
                print(f"📍 Filtering by location: {location_normalized}")
        
        # Price range filter
        if min_price is not None or max_price is not None:
            price_range = {}
            if min_price is not None:
                price_range["gte"] = min_price
            if max_price is not None:
                price_range["lte"] = max_price
            
            conditions.append(
                FieldCondition(
                    key="price",
                    range=Range(**price_range)
                )
            )
        
        # Additional filters from kwargs
        for key, value in kwargs.items():
            if value is not None:
                if isinstance(value, (int, float)):
                    conditions.append(
                        FieldCondition(
                            key=key,
                            range=Range(gte=value, lte=value)
                        )
                    )
                else:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=str(value))
                        )
                    )
        
        if conditions:
            return Filter(must=conditions)
        return None
    
    def search_similar(self, 
                      query: str,
                      location: Optional[str] = None,
                      min_price: Optional[float] = None,
                      max_price: Optional[float] = None,
                      top_k: int = 5,
                      **kwargs) -> List[Dict]:
        """
        Search for similar experiences using vector similarity and constraints.
        
        Args:
            query: User's search query
            location: Location constraint
            min_price: Minimum price constraint
            max_price: Maximum price constraint
            top_k: Number of results to return
            **kwargs: Additional filter conditions
            
        Returns:
            List of search results with similarity scores
        """
        try:
            # Parse query to extract tags
            query_tags = self.parse_user_query(query)
            print(f"Extracted tags from query: {query_tags}")
            
            if not query_tags:
                print("⚠️ No valid tags found in query.")
                print(f"Available tags: {', '.join(self.all_tags)}")
                return []
            
            # Create query vector
            query_vector = self.create_query_vector(query_tags)
            
            # Create filters for constraints
            filters = self.create_filters(
                location=location,
                min_price=min_price,
                max_price=max_price,
                **kwargs
            )
            
            # Perform vector search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=filters,
                limit=top_k,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Format results
            results = []
            for result in search_results:
                result_dict = {
                    "id": result.id,
                    "similarity_score": float(result.score),
                    "payload": result.payload
                }
                results.append(result_dict)
            
            return results
            
        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            return []
    
    def print_results(self, results: List[Dict], query: str):
        """
        Print search results in a formatted way.
        
        Args:
            results: Search results
            query: Original query
        """
        if not results:
            print("\n❌ No results found.")
            return
        
        print(f"\n🔍 Search Results for: '{query}'")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            payload = result["payload"]
            score = result["similarity_score"]
            
            print(f"\n{i}. Similarity Score: {score:.4f}")
            print(f"   Name: {payload.get('name', 'N/A')}")
            print(f"   Tags: {payload.get('tags', 'N/A')}")
            print(f"   Location: {payload.get('location', 'N/A')}")
            print(f"   Price: ${payload.get('price', 'N/A')}")
            
            # Print additional fields if available
            for key, value in payload.items():
                if key not in ['name', 'tags', 'location', 'price', 'original_index']:
                    print(f"   {key.replace('_', ' ').title()}: {value}")
            
            print("-" * 60)
    
    def get_available_tags(self) -> List[str]:
        """Get list of all available tags."""
        return self.all_tags.copy()
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "status": info.status,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "config": info.config
            }
        except Exception as e:
            return {"error": str(e)}

    def set_semantic_threshold(self, threshold: float):
        """
        Set the similarity threshold for semantic tag matching.
        
        Args:
            threshold: Similarity threshold (0.0 to 1.0)
        """
        self.semantic_threshold = max(0.0, min(1.0, threshold))
        print(f"🎯 Semantic similarity threshold set to: {self.semantic_threshold}")
    
    def analyze_query_semantics(self, query: str, top_k: int = 10) -> List[tuple]:
        """
        Analyze semantic similarity between query and all tags for debugging.
        
        Args:
            query: User's search query
            top_k: Number of top matches to return
            
        Returns:
            List of (tag, similarity_score) tuples
        """
        query_embedding = self.semantic_model.encode([query])
        tag_embeddings = self.semantic_model.encode(self.all_tags)
        
        similarities = torch.nn.functional.cosine_similarity(
            torch.tensor(query_embedding), 
            torch.tensor(tag_embeddings), 
            dim=1
        )
        
        tag_scores = [(self.all_tags[i], similarities[i].item()) 
                     for i in range(len(self.all_tags))]
        tag_scores.sort(key=lambda x: x[1], reverse=True)
        
        return tag_scores[:top_k]

    def get_allowed_locations(self) -> List[str]:
        """Get list of allowed locations."""
        return ["Delhi", "Bangalore", "Gurugram"]
    
    def validate_location(self, location: str) -> Optional[str]:
        """
        Validate and normalize location input using intelligent parsing.
        
        Args:
            location: User input location
            
        Returns:
            Normalized location name (always returns a valid location)
        """
        if not location:
            return None
            
        return self.normalize_location(location)

    def normalize_location(self, location_string: str) -> str:
        """
        Parse and normalize location string to match allowed cities.
        Uses the same intelligent parsing as the uploader.
        
        Args:
            location_string: Raw location string from user input
            
        Returns:
            Normalized location (Delhi, Bangalore, or Gurugram)
        """
        if not location_string:
            return 'Delhi'  # Default fallback
        
        location_lower = str(location_string).lower().strip()
        
        # Define mapping patterns for each city
        city_patterns = {
            'Delhi': [
                'delhi', 'new delhi', 'old delhi', 'ncr', 'national capital region',
                'dwarka', 'rohini', 'lajpat nagar', 'connaught place', 'cp'
            ],
            'Bangalore': [
                'bangalore', 'bengaluru', 'bangaluru', 'blr', 'karnataka',
                'electronic city', 'whitefield', 'koramangala', 'indiranagar',
                'jayanagar', 'mg road', 'brigade road'
            ],
            'Gurugram': [
                'gurugram', 'gurgaon', 'cyber city', 'dlf', 'millennium city',
                'haryana', 'manesar', 'sohna', 'udyog vihar', 'golf course road'
            ]
        }
        
        # Check for exact or partial matches
        for city, patterns in city_patterns.items():
            for pattern in patterns:
                if pattern in location_lower:
                    return city
        
        # If no match found, try fuzzy matching on city names only
        cities = ['delhi', 'bangalore', 'bengaluru', 'gurugram', 'gurgaon']
        for city_name in cities:
            if city_name in location_lower:
                if city_name in ['bangalore', 'bengaluru']:
                    return 'Bangalore'
                elif city_name in ['gurugram', 'gurgaon']:
                    return 'Gurugram'
                elif city_name == 'delhi':
                    return 'Delhi'
        
        # Final fallback to Delhi
        print(f"⚠️ Could not parse location '{location_string}', defaulting to Delhi")
        return 'Delhi'

def main():
    """Main function for interactive search."""
    try:
        print("🌐 Qdrant Cloud Experience Search Engine")
        print("Make sure you have uploaded data using 'qdrant_uploader.py' first!")
        
        # Check for environment variables first
        url = os.getenv('QDRANT_URL')
        api_key = os.getenv('QDRANT_API_KEY')
        
        if not url or not api_key:
            print("\n📝 Environment variables not found.")
            print("You can either:")
            print("1. Create a .env file with QDRANT_URL and QDRANT_API_KEY")
            print("2. Enter them manually below")
            
            url = url or input("\nEnter Qdrant Cloud URL: ").strip()
            api_key = api_key or input("Enter Qdrant Cloud API Key: ").strip()
            
            if not url or not api_key:
                print("❌ Both URL and API Key are required!")
                return
        
        search_engine = QdrantSearchEngine(url=url, api_key=api_key)
        
        print("\nAvailable tags:", ', '.join(search_engine.get_available_tags()[:15]), "...")
        print("Available locations:", ', '.join(search_engine.get_allowed_locations()))
        
        # Example searches with semantic understanding
        examples = [
            {
                "query": "something thrilling and exciting outdoors",
                "location": None,
                "max_price": 200,
                "description": "Semantic: Thrilling outdoor activities under $200"
            },
            {
                "query": "intimate dinner for couples",
                "location": "Delhi",
                "max_price": None,
                "description": "Semantic: Romantic experiences in Delhi"
            },
            {
                "query": "museum visits and cultural learning",
                "location": "Bangalore",
                "max_price": 50,
                "description": "Semantic: Cultural experiences in Bangalore under $50"
            },
            {
                "query": "peaceful and calming spa treatments",
                "location": "Gurugram",
                "min_price": 100,
                "description": "Semantic: Premium wellness experiences in Gurugram"
            }
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"\n{'='*80}")
            print(f"EXAMPLE {i}: {example['description']}")
            print(f"{'='*80}")
            
            results = search_engine.search_similar(
                query=example["query"],
                location=example["location"],
                min_price=example.get("min_price"),
                max_price=example.get("max_price"),
                top_k=3
            )
            
            search_engine.print_results(results, example["query"])
        
        # Interactive search
        print(f"\n{'='*80}")
        print("INTERACTIVE SEMANTIC SEARCH")
        print(f"{'='*80}")
        print("🤖 This search engine now understands the MEANING of your queries!")
        print("Try queries like:")
        print("  • 'something thrilling and exciting'")
        print("  • 'peaceful relaxing activities'") 
        print("  • 'educational cultural experiences'")
        print("  • 'romantic activities for couples'")
        print("\nEnter your search queries (type 'quit' to exit, 'analyze' to see semantic breakdown):")
        
        while True:
            query = input("\n🔍 Enter search query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query.lower() == 'analyze':
                analyze_query = input("🔍 Enter query to analyze: ").strip()
                if analyze_query:
                    print(f"\n📊 Semantic Analysis for: '{analyze_query}'")
                    print("-" * 50)
                    analysis = search_engine.analyze_query_semantics(analyze_query, top_k=10)
                    for tag, score in analysis:
                        print(f"   {tag}: {score:.4f}")
                continue
            
            if not query:
                continue
            
            # Optional constraints
            location = input("📍 Location (Delhi/Bangalore/Gurugram, optional): ").strip() or None
            
            try:
                max_price_input = input("💰 Max price (optional): ").strip()
                max_price = float(max_price_input) if max_price_input else None
            except ValueError:
                max_price = None
            
            results = search_engine.search_similar(
                query=query,
                location=location,
                max_price=max_price,
                top_k=5
            )
            
            search_engine.print_results(results, query)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure you have:")
        print("1. Uploaded data using 'qdrant_uploader.py'")
        print("2. Correct Qdrant Cloud credentials")
        print("3. Stable internet connection")

if __name__ == "__main__":
    main()

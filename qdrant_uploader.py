import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Optional
import uuid
import json
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Supabase Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
SUPABASE_TABLE_NAME = os.getenv('SUPABASE_TABLE_NAME')


class QdrantDataUploader:
    def __init__(self, url: str = None, api_key: str = None, collection_name: str = None):
        """
        Initialize Qdrant Cloud client and data uploader.
        
        Args:
            url: Qdrant Cloud URL
            api_key: Qdrant Cloud API key
            collection_name: Name of the Qdrant collection
        """
        # Get credentials from parameters or environment variables
        cloud_url = url or os.getenv('QDRANT_URL') or os.getenv('QDRANT_CLOUD_URL')
        cloud_api_key = api_key or os.getenv('QDRANT_API_KEY') or os.getenv('QDRANT_CLOUD_API_KEY')
        collection_name = collection_name or os.getenv('QDRANT_COLLECTION_NAME') or "recbot_v1"
        
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
        
        # Initialize Supabase client
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print(f"📊 Connected to Supabase: {SUPABASE_URL}")
        
        self.collection_name = collection_name
        self.all_tags = []
        self.tag_to_index = {}
    
    def fetch_supabase_data(self) -> List[Dict]:
        """
        Fetch experience data from Supabase table.
        
        Returns:
            List of experience records
        """
        try:
            print(f"📥 Fetching data from Supabase table: {SUPABASE_TABLE_NAME}")
            
            # Fetch all records from the experience table
            response = self.supabase.table(SUPABASE_TABLE_NAME).select("*").execute()
            
            if response.data:
                print(f"✅ Fetched {len(response.data)} records from Supabase")
                
                # Process and normalize locations
                filtered_data = []
                location_stats = {'Delhi': 0, 'Bangalore': 0, 'Gurugram': 0}
                
                for record in response.data:
                    # Use intelligent location parsing
                    original_location = record.get('location', '')
                    normalized_location = self.normalize_location(original_location)
                    
                    # Update the record with normalized location
                    record['location'] = normalized_location
                    record['original_location'] = original_location  # Keep original for reference
                    
                    # Track statistics
                    location_stats[normalized_location] += 1
                    
                    if original_location.strip() and original_location.strip().lower() != normalized_location.lower():
                        print(f"📍 Mapped '{original_location}' → '{normalized_location}' for record {record.get('id', 'N/A')}")
                    
                    filtered_data.append(record)
                
                # Print location distribution summary
                print(f"📊 Location distribution after normalization:")
                for city, count in location_stats.items():
                    print(f"   • {city}: {count} records")
                
                return filtered_data
            else:
                print("❌ No data found in Supabase table")
                return []
                
        except Exception as e:
            print(f"❌ Error fetching Supabase data: {str(e)}")
            return []
    
    def process_supabase_data(self, data: List[Dict]) -> tuple:
        """
        Process Supabase data and extract unique tags.
        
        Args:
            data: List of records from Supabase
            
        Returns:
            Tuple of (processed_data, all_tags, tag_to_index)
        """
        all_tags = set()
        
        # Extract all unique tags from the data
        for record in data:
            tags_field = record.get('tags', '')
            if tags_field:
                tags = [tag.strip().lower() for tag in str(tags_field).split(',')]
                all_tags.update(tags)
        
        # Sort tags for consistent indexing
        self.all_tags = sorted(list(all_tags))
        self.tag_to_index = {tag: idx for idx, tag in enumerate(self.all_tags)}
        
        print(f"Found {len(self.all_tags)} unique tags: {self.all_tags}")
        return data, self.all_tags, self.tag_to_index
        
    def process_csv_data(self, csv_file_path: str) -> tuple:
        """
        Process CSV data and extract unique tags.
        
        Args:
            csv_file_path: Path to CSV file
            
        Returns:
            Tuple of (dataframe, all_tags, tag_to_index)
        """
        df = pd.read_csv(csv_file_path)
        all_tags = set()
        
        # Extract all unique tags
        for _, row in df.iterrows():
            if pd.notna(row.get('tags', '')):
                tags = [tag.strip().lower() for tag in str(row['tags']).split(',')]
                all_tags.update(tags)
        
        # Sort tags for consistent indexing
        self.all_tags = sorted(list(all_tags))
        self.tag_to_index = {tag: idx for idx, tag in enumerate(self.all_tags)}
        
        print(f"Found {len(self.all_tags)} unique tags: {self.all_tags}")
        return df, self.all_tags, self.tag_to_index
    
    def create_binary_vector(self, tags_string: str) -> List[float]:
        """
        Convert tags string to binary vector.
        
        Args:
            tags_string: Comma-separated string of tags
            
        Returns:
            Binary vector as list of floats
        """
        vector = [0.0] * len(self.all_tags)
        
        if pd.notna(tags_string) and tags_string:
            tags = [tag.strip().lower() for tag in str(tags_string).split(',')]
            for tag in tags:
                if tag in self.tag_to_index:
                    vector[self.tag_to_index[tag]] = 1.0
        
        return vector
    
    def create_collection(self):
        """Create Qdrant collection with appropriate vector configuration."""
        try:
            # Delete collection if it exists
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        except:
            print(f"Collection {self.collection_name} doesn't exist, creating new one")
        
        # Create new collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=len(self.all_tags),  # Vector dimension = number of tags
                distance=Distance.COSINE  # Use cosine distance for similarity
            )
        )
        print(f"Created collection '{self.collection_name}' with {len(self.all_tags)} dimensions")
    
    def upload_data(self, csv_file_path: str) -> bool:
        """
        Upload CSV data to Qdrant vector database.
        
        Args:
            csv_file_path: Path to CSV file
            
        Returns:
            Success status
        """
        try:
            # Process CSV data
            df, all_tags, tag_to_index = self.process_csv_data(csv_file_path)
            
            # Create collection
            self.create_collection()
            
            # Prepare points for upload
            points = []
            for idx, row in df.iterrows():
                # Create binary vector from tags
                vector = self.create_binary_vector(row.get('tags', ''))
                
                # Create payload with all row data
                payload = {
                    "name": str(row.get('name', '')),
                    "tags": str(row.get('tags', '')),
                    "location": str(row.get('location', '')),
                    "price": float(row.get('price', 0)),
                    "original_index": int(idx)
                }
                
                # Add any additional columns
                for col in df.columns:
                    if col not in ['name', 'tags', 'location', 'price']:
                        payload[col] = str(row[col]) if pd.notna(row[col]) else ""
                
                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),  # Generate unique ID
                    vector=vector,
                    payload=payload
                )
                points.append(point)
            
            # Upload points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                print(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
            
            # Save metadata for search module
            metadata = {
                "all_tags": self.all_tags,
                "tag_to_index": self.tag_to_index,
                "total_items": len(df),
                "vector_dimension": len(self.all_tags)
            }
            
            with open("qdrant_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Successfully uploaded {len(points)} items to Qdrant")
            print(f"Metadata saved to 'qdrant_metadata.json'")
            return True
            
        except Exception as e:
            print(f"Error uploading data: {str(e)}")
            return False
    
    def upload_supabase_data(self) -> bool:
        """
        Upload Supabase data to Qdrant vector database.
        
        Returns:
            Success status
        """
        try:
            # Fetch data from Supabase
            supabase_data = self.fetch_supabase_data()
            
            if not supabase_data:
                print("❌ No data to upload")
                return False
            
            # Process Supabase data
            data, all_tags, tag_to_index = self.process_supabase_data(supabase_data)
            
            # Create collection
            self.create_collection()
            
            # Prepare points for upload
            points = []
            for record in data:
                # Create binary vector from tags
                vector = self.create_binary_vector(record.get('tags', ''))
                
                # Create payload with record data
                payload = {
                    "id": str(record.get('id', '')),
                    "name": str(record.get('name', '')),
                    "tags": str(record.get('tags', '')),
                    "location": str(record.get('location', '')),
                    "price": float(record.get('price', 0))
                }
                
                # Add any additional fields from Supabase
                for key, value in record.items():
                    if key not in ['id', 'name', 'tags', 'location', 'price']:
                        payload[key] = str(value) if value is not None else ""
                
                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),  # Generate unique ID for Qdrant
                    vector=vector,
                    payload=payload
                )
                points.append(point)
            
            # Upload points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                print(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
            
            # Save metadata for search module
            metadata = {
                "all_tags": self.all_tags,
                "tag_to_index": self.tag_to_index,
                "total_items": len(data),
                "vector_dimension": len(self.all_tags),
                "data_source": "supabase",
                "table_name": SUPABASE_TABLE_NAME
            }
            
            with open("qdrant_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Successfully uploaded {len(points)} items from Supabase to Qdrant")
            print(f"Metadata saved to 'qdrant_metadata.json'")
            return True
            
        except Exception as e:
            print(f"Error uploading Supabase data: {str(e)}")
            return False

    def create_sample_data(self, filename: str = "sample_experiences.csv") -> str:
        """Create sample experience data for testing."""
        sample_data = {
            'name': [
                'Skydiving Adventure', 'Art Gallery Tour', 'Romantic Dinner Cruise', 
                'Stand-up Comedy Show', 'Beach Yoga Retreat', 'Mountain Climbing Expedition',
                'Shopping District Tour', 'Historic Castle Visit', 'Jazz Club Night', 'Luxury Spa Day',
                'Food Truck Festival', 'Wine Tasting Experience', 'Motorcycle Tour', 'Photography Workshop',
                'Dance Class', 'Cooking Class', 'Museum Tour', 'Concert Experience', 'Gaming Cafe',
                'Nature Hiking Trail'
            ],
            'tags': [
                'adventure,extreme,outdoor,adrenaline', 'art,culture,indoor,educational', 'romantic,dining,water,luxury',
                'comedy,entertainment,indoor,social', 'beach,relaxing,wellness,outdoor', 'adventure,outdoor,nature,extreme',
                'shopping,indoor,fashion,leisure', 'historic,culture,art,educational', 'nightlife,music,entertainment,social', 'relaxing,spa,luxury,wellness',
                'food,outdoor,social,cultural', 'wine,luxury,educational,social', 'adventure,outdoor,transportation,exciting',
                'art,educational,creative,indoor', 'dance,social,entertainment,active', 'cooking,educational,indoor,cultural',
                'culture,educational,art,indoor', 'music,entertainment,social,outdoor', 'gaming,indoor,social,entertainment',
                'nature,outdoor,hiking,wellness'
            ],
            'location': [
                'New York', 'Paris', 'Venice', 'London', 'Miami', 'Denver',
                'Tokyo', 'Prague', 'New Orleans', 'Bali', 'Austin', 'Napa Valley',
                'Los Angeles', 'Florence', 'Barcelona', 'Rome', 'Berlin', 'Nashville',
                'Seoul', 'Vancouver'
            ],
            'price': [300, 45, 180, 25, 150, 250, 60, 30, 40, 400, 35, 120, 200, 80, 50, 90, 25, 75, 20, 40],
            'duration_hours': [4, 2, 3, 1.5, 6, 8, 4, 2, 3, 4, 5, 3, 6, 4, 2, 3, 2, 4, 3, 5],
            'min_age': [18, 0, 21, 18, 16, 18, 0, 0, 21, 16, 0, 21, 18, 12, 12, 16, 0, 0, 12, 12],
            'difficulty': ['Hard', 'Easy', 'Easy', 'Easy', 'Medium', 'Hard', 'Easy', 'Easy', 'Easy', 'Easy',
                          'Easy', 'Easy', 'Medium', 'Easy', 'Medium', 'Medium', 'Easy', 'Easy', 'Easy', 'Medium']
        }
        
        df_sample = pd.DataFrame(sample_data)
        df_sample.to_csv(filename, index=False)
        print(f"Sample experience data '{filename}' created successfully!")
        return filename

    def test_supabase_connection(self) -> bool:
        """
        Test connection to Supabase and verify table structure.
        
        Returns:
            Connection success status
        """
        try:
            print("🔍 Testing Supabase connection...")
            
            # Test basic connection
            response = self.supabase.table(SUPABASE_TABLE_NAME).select("*").limit(1).execute()
            
            if response.data:
                print("✅ Supabase connection successful!")
                
                # Check required fields
                required_fields = ['id', 'tags', 'location', 'price']
                sample_record = response.data[0]
                missing_fields = [field for field in required_fields if field not in sample_record]
                
                if missing_fields:
                    print(f"⚠️ Missing required fields in table: {missing_fields}")
                    return False
                
                print(f"✅ Table structure verified. Required fields present: {required_fields}")
                print(f"📊 Sample record fields: {list(sample_record.keys())}")
                return True
            else:
                print("❌ Table is empty or doesn't exist")
                return False
                
        except Exception as e:
            print(f"❌ Supabase connection failed: {str(e)}")
            return False

    def normalize_location(self, location_string: str) -> str:
        """
        Parse and normalize location string to match allowed cities.
        
        Args:
            location_string: Raw location string from Supabase
            
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

    def show_location_mapping_examples(self):
        """
        Display examples of how location strings will be mapped.
        """
        print("\n📍 Location Mapping Examples:")
        examples = [
            "New Delhi", "Bangalore", "Gurgaon", "Electronic City",
            "Cyber City", "Koramangala", "Dwarka", "Whitefield",
            "DLF Phase 1", "Indiranagar", "Rohini", "MG Road",
            "Haryana", "Karnataka", "NCR", "Unknown City"
        ]
        
        for example in examples:
            mapped = self.normalize_location(example)
            print(f"   '{example}' → {mapped}")

def main():
    """Main function to demonstrate Supabase data upload."""
    
    # Load configuration from environment
    supabase_url = os.getenv('SUPABASE_URL') or "https://ceqpdprcqhmkqdbgmmkn.supabase.co"
    supabase_table = os.getenv('SUPABASE_TABLE_NAME') or "experience"
    
    print("🌐 Qdrant Cloud Data Uploader (Supabase Integration)")
    print("Fetching data from Supabase and uploading to Qdrant Cloud...")
    print(f"📊 Supabase URL: {supabase_url}")
    print(f"📋 Table: {supabase_table}")
    
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
    
    try:
        uploader = QdrantDataUploader(url=url, api_key=api_key)
        
        # Show location mapping examples
        uploader.show_location_mapping_examples()
        
        # Test Supabase connection
        if not uploader.test_supabase_connection():
            print("❌ Supabase connection test failed!")
            return
        
        # Upload data from Supabase to Qdrant Cloud
        success = uploader.upload_supabase_data()
        
        if success:
            print("\n✅ Supabase data upload completed successfully!")
            print("📍 Locations normalized to: Delhi, Bangalore, Gurugram")
            print("You can now use 'qdrant_search.py' to perform similarity searches")
        else:
            print("\n❌ Data upload failed!")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please check your Qdrant Cloud credentials and Supabase connection.")
        
    # Optional: Create sample CSV for backup/testing
    print("\n💾 Creating backup CSV file from uploaded data...")
    try:
        uploader = QdrantDataUploader(url=url, api_key=api_key)
        supabase_data = uploader.fetch_supabase_data()
        if supabase_data:
            df = pd.DataFrame(supabase_data)
            df.to_csv("supabase_backup.csv", index=False)
            print("✅ Backup CSV created: 'supabase_backup.csv'")
    except Exception as e:
        print(f"⚠️ Could not create backup CSV: {str(e)}")

if __name__ == "__main__":
    main()

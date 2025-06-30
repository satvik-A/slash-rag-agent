# Qdrant Experience Recommendation System

A vector-based recommendation system using Qdrant that converts experience tags to binary vectors and performs cosine similarity search with location and price constraints. **Now featuring semantic understanding** that can interpret the meaning of your queries!

## 🚀 Features

- **🧠 Semantic Query Understanding**: AI-powered query interpretation that understands meaning, not just keywords
- **Binary Vector Encoding**: Tags converted to binary vectors (e.g., `adventure,outdoor,art` → `[1,1,0,...]`)
- **Vector Database**: Qdrant for efficient similarity search
- **Hard Constraints**: Location and price filtering before similarity calculation
- **Cosine Similarity**: Accurate matching based on tag overlap
- **Scalable**: Handles large datasets efficiently

## 🤖 Semantic Search Examples

The system now understands natural language queries:

```python
# Traditional keyword search
"adventure outdoor extreme"

# New semantic search - understands meaning!
"something thrilling and exciting outdoors"
"peaceful relaxing activities" 
"educational cultural experiences"
"romantic activities for couples"
"fun family-friendly activities"
```

The AI model can map concepts like:
- "thrilling" → `adventure`, `extreme`
- "peaceful" → `relaxing`, `wellness`, `spa`
- "couples" → `romantic`, `dining`, `luxury`
- "educational" → `culture`, `art`, `museum`

## 📋 Prerequisites

1. **Python 3.7+**
2. **Qdrant Cloud Account**: [cloud.qdrant.io](https://cloud.qdrant.io/) - Free tier available

## 🛠️ Installation

### Quick Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Sign up at https://cloud.qdrant.io/
# 3. Create cluster and get URL + API key
# 4. Copy .env.template to .env and add your credentials

# 5. Upload data
python qdrant_uploader.py

# 6. Search experiences
python qdrant_search.py
```

## 📊 Data Format

Your CSV should have these columns:
```csv
name,tags,location,price
"Skydiving Adventure","adventure,extreme,outdoor","New York",300
"Art Gallery Tour","art,culture,indoor","Paris",45
"Romantic Dinner","romantic,dining,luxury","Venice",180
```

Required columns:
- `tags`: Comma-separated tags (converted to binary vectors)
- `location`: Location for filtering
- `price`: Price for range filtering
- `name`: Display name

Optional columns will be included in search results.

## 🔧 Usage

### 1. Upload Your Data

```python
from qdrant_uploader import QdrantDataUploader

# Using environment variables (.env file)
uploader = QdrantDataUploader()
uploader.upload_data("your_experiences.csv")

# Using direct parameters
uploader = QdrantDataUploader(
    url="https://your-cluster.qdrant.cloud",
    api_key="your-api-key"
)
uploader.upload_data("your_experiences.csv")
```

### 2. Search Experiences
```python
from qdrant_search import QdrantSearchEngine

# Using environment variables (.env file)
search_engine = QdrantSearchEngine()

# Using direct parameters
search_engine = QdrantSearchEngine(
    url="https://your-cluster.qdrant.cloud",
    api_key="your-api-key"
)

# Basic search
results = search_engine.search_similar("adventure outdoor activities")

# With constraints
results = search_engine.search_similar(
    query="romantic luxury dining",
    location="Venice",
    max_price=200,
    top_k=5
)

# Print results
search_engine.print_results(results, "romantic luxury dining")
```

### 3. Interactive Search
```bash
python qdrant_search.py
```

## 🏗️ System Architecture

### Vector Encoding Process:
1. **Tag Extraction**: Extract all unique tags from dataset
2. **Index Mapping**: Create tag → index mapping
3. **Binary Vectors**: Convert tags to binary vectors where each dimension represents a tag
4. **Upload**: Store vectors in Qdrant with metadata

### Search Process:
1. **🧠 Semantic Analysis**: AI model understands query meaning and maps to relevant tags
2. **Fallback Matching**: If no semantic matches, falls back to keyword matching
3. **Vector Creation**: Convert identified tags to binary vector
4. **Constraint Filtering**: Apply location/price filters
5. **Similarity Search**: Cosine similarity in vector space
6. **Result Ranking**: Return top-K most similar items

## 📁 File Structure

```
├── qdrant_uploader.py      # Data upload to Qdrant
├── qdrant_search.py        # Search and similarity engine  
├── setup.py               # Automated setup script
├── requirements.txt       # Python dependencies
├── experiences.csv        # Sample data (auto-generated)
├── qdrant_metadata.json   # Tag metadata (auto-generated)
└── README.md             # This file
```

## 🔍 Example Searches

```python
# Semantic search - understands meaning!
results = search_engine.search_similar(
    "something thrilling and exciting outdoors", 
    max_price=200
)

# Traditional approach still works
results = search_engine.search_similar(
    "art culture museum", 
    location="Paris"
)

# Natural language queries
results = search_engine.search_similar(
    "peaceful spa treatments for wellness", 
    min_price=100
)

# Analyze semantic understanding
analysis = search_engine.analyze_query_semantics("romantic dinner for two")
# Shows which tags the AI thinks match your query and with what confidence
```

## 🎯 Semantic Understanding Example

When you search for **"something thrilling and exciting outdoors"**:

```
🔍 Analyzing query semantically: 'something thrilling and exciting outdoors'
📊 Semantic matches found:
   • adventure: 0.742
   • extreme: 0.681
   • outdoor: 0.895
   • exciting: 0.734
```

The AI understands that:
- "thrilling" semantically matches `adventure` and `extreme`
- "exciting" maps to `exciting` and `adventure`
- "outdoors" directly matches `outdoor`

## 🎯 Vector Similarity Example

If available tags are: `['adventure', 'art', 'culture', 'outdoor', 'romantic']`

Then:
- `"adventure,outdoor"` → `[1, 0, 0, 1, 0]`
- `"art,culture"` → `[0, 1, 1, 0, 0]`  
- `"romantic,outdoor"` → `[0, 0, 0, 1, 1]`

Cosine similarity between vectors determines match strength.

## 🐛 Troubleshooting

### Qdrant Cloud Connection Issues
- Verify your cluster URL format: `https://your-cluster.qdrant.cloud`
- Check your API key is correct
- Ensure stable internet connection
- Verify your Qdrant Cloud cluster is running

### No Search Results
- Check if data was uploaded: `python qdrant_uploader.py`
- Verify tag names match available tags
- Check constraint filters (location/price)

### Package Installation Issues
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Authentication Errors
- Double-check your Qdrant Cloud credentials
- Make sure your API key hasn't expired
- Verify you have access to the cluster

## 🔧 Customization

### Add New Constraints
```python
# In qdrant_search.py, modify create_filters method
def create_filters(self, min_age=None, difficulty=None, **kwargs):
    # Add custom filters
    if min_age:
        conditions.append(FieldCondition(key="min_age", range=Range(lte=min_age)))
```

### Modify Vector Encoding
```python
# In qdrant_uploader.py, modify create_binary_vector method
def create_binary_vector(self, tags_string: str) -> List[float]:
    # Add weighted vectors, TF-IDF, etc.
    pass
```

## 📈 Performance Tips

1. **Batch Uploads**: Upload in batches of 100-1000 items
2. **Index Optimization**: Qdrant automatically optimizes indexes
3. **Filter First**: Use constraints to reduce search space
4. **Vector Dimensions**: Keep tag count reasonable (<1000 dimensions)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

- **Documentation**: [Qdrant Docs](https://qdrant.tech/documentation/)
- **Issues**: Create GitHub issue
- **Discord**: [Qdrant Community](https://discord.gg/qdrant)

## 🧠 Semantic Search Configuration

```python
# Adjust semantic similarity threshold
search_engine.set_semantic_threshold(0.4)  # Higher = more strict matching

# Analyze how the AI interprets your query
analysis = search_engine.analyze_query_semantics("fun family activities")
for tag, score in analysis:
    print(f"{tag}: {score:.3f}")

# Use the analyze command in interactive mode
# Type 'analyze' then enter your query to see semantic breakdown
```

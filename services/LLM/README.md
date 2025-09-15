# Global LLM Service

This module provides a global LLM service instance that can be accessed from anywhere in the application.

## Usage Examples

### Basic Import and Usage

```python
# Import the global LLM service
from services.LLM import llm_service, chat_completion, get_llm_service

# Check if LLM is available
from services.LLM import is_llm_available
if is_llm_available():
    print("LLM service is ready!")
```

### Chat Completion

```python
from services.LLM import chat_completion

# Simple chat completion
messages = [
    {"role": "system", "content": "You are a helpful legal assistant."},
    {"role": "user", "content": "What is contract law?"}
]

response = await chat_completion(messages, is_legal_query=True)
print(response)
```

### Using the Service Instance Directly

```python
from services.LLM import get_llm_service

llm = get_llm_service()

# Get configuration info
config = llm.get_config_info()
print(f"Using model: {config['model']}")

# Custom chat completion with parameters
response = await llm.chat_completion(
    messages,
    temperature=0.1,
    max_tokens=1000,
    model="gpt-4"
)
```

### Generate Embeddings

```python
from services.LLM import generate_embeddings

embeddings = await generate_embeddings("Legal document text here")
print(f"Generated {len(embeddings)} dimensional embedding")
```

### Access Raw OpenAI Client

```python
from services.LLM import get_llm_client

client = get_llm_client()
if client:
    # Use the raw OpenAI client for advanced operations
    response = await client.chat.completions.create(...)
```

## Configuration

The service automatically uses settings from `core.config.settings`:

- `LLM_PROVIDER`: "openai" or "azure_openai"
- `LLM_MODEL`: Default model to use
- `OPENAI_API_KEY`: OpenAI API key
- `AZURE_OPENAI_*`: Azure OpenAI configuration
- Temperature, token limits, and other behavior settings

## Features

- **Singleton Pattern**: One global instance shared across the application
- **Automatic Initialization**: Service initializes when first imported
- **Error Handling**: Graceful handling of missing API keys or network issues
- **Multiple Providers**: Support for OpenAI and Azure OpenAI
- **Convenience Functions**: Simple functions for common operations
- **Configuration Info**: Easy access to current settings and status

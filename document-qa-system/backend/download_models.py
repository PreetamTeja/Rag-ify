# Download embedding model
from sentence_transformers import SentenceTransformer
print("Downloading embedding model...")
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
print("Embedding model ready!")

# Download LLM model
from transformers import AutoTokenizer, AutoModelForCausalLM
print("Downloading LLM model...")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
print("LLM model ready!")

# Exit Python
exit()
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv("./.env", override=True)

class QdrantSettings(BaseSettings):
    QDRANT_URL: str
    QDRANT_API_KEY: str

    GEMINI_EMBEDDING_MODEL_NAME: str
    GEMINI_API_KEY: str

qdrant_settings = QdrantSettings()

embeddings = GoogleGenerativeAIEmbeddings(
    model=qdrant_settings.GEMINI_EMBEDDING_MODEL_NAME,
    google_api_key=qdrant_settings.GEMINI_API_KEY,
)

loader = PyPDFLoader("./master_direction.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(documents)

# Reduce batch size and add retry logic
batch_size = 10
max_retries = 3

logger.info("Initializing Qdrant Vector Store...")

for i in range(0, len(split_documents), batch_size):
    batch = split_documents[i:i + batch_size]
    for attempt in range(max_retries):
        try:
            logger.info(f"Processing batch {i // batch_size + 1} (attempt {attempt + 1})...")
            vector_store = QdrantVectorStore.from_documents(
                documents=batch,
                url=qdrant_settings.QDRANT_URL,
                api_key=qdrant_settings.QDRANT_API_KEY,
                embedding=embeddings,
                collection_name="financial_products_collection"
            )
            logger.info(f"Batch {i // batch_size + 1} processed successfully.")
            break
        except Exception as e:
            logger.error(f"Error processing batch {i // batch_size + 1} on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to process batch {i // batch_size + 1} after {max_retries} attempts.")
                raise

logger.info("Qdrant Vector Store initialized with documents.")
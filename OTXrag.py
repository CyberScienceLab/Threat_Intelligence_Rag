from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import VectorParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import csv
import time
import os
from typing import List, Dict, Tuple
from transformers.utils import logging
logging.set_verbosity_error()

#####################################################################
# Constants and Configuration

# Configuration constants for Qdrant and other settings.
QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION_NAME = "OTX_Cyber_Threat_Intelligence"
DEFAULT_VECTOR_SIZE = 1024
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
BATCH_SIZE = 100
EMBEDDING_DELAY_SECONDS = 5

#####################################################################
# Utility Functions for Qdrant

class QdrantUtility:
    """Utility class for managing Qdrant operations such as creating collections, retrieving content, and loading embeddings."""
    def __init__(self, collection_name: str = DEFAULT_COLLECTION_NAME):
        # Initialize the QdrantUtility with the specified or default collection name.
        self.collection_name = collection_name
        # Initialize embeddings using HuggingFace's BigGraph Embeddings (BGE) model.
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en",
            model_kwargs={'device': 'cpu'}, # Specify model to run on CPU.
            encode_kwargs={'normalize_embeddings': False} # Do not normalize embeddings.
        )

    def create_collection(self, collection_name: str = DEFAULT_COLLECTION_NAME, vector_size: int = DEFAULT_VECTOR_SIZE):
        """Create a collection in Qdrant if it doesn't already exist."""
        try:
            client = QdrantClient(url=QDRANT_URL, prefer_grpc=False) # Initialize Qdrant client.
            # Create a collection with specified vector size and cosine distance metric.
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance='Cosine'),
            )
            print(f"Successfully created collection: {collection_name}")

        except Exception as e:
            print(f"Error creating collection {collection_name}: {e}") # Print error if collection creation fails.
        
        finally: 
            client.close() # Ensure the client connection is closed.
    
    def retrieve_relevant_content(self, query: str, num_documents: int) -> List[str]:
        """Retrieve content from Qdrant that is relevant to the given query."""
        client = QdrantClient(url=QDRANT_URL, prefer_grpc=False) # Initialize Qdrant client.

        # Use Qdrant to perform a similarity search using the embeddings.
        db = Qdrant(client=client, embeddings=self.embeddings, collection_name=self.collection_name)
        docs = db.similarity_search_with_score(query=query, k=num_documents)
        # Extract content from the search results.
        content = [f"context[{doc.page_content}/n{doc.metadata}]" for doc, _ in docs]

        client.close() # Close the client connection.
        return content 

    def load_embeddings_with_custom_metadata(self, texts: List[str], metadata: List[Dict]):
        """Load embeddings with custom metadata into Qdrant."""
        client = QdrantClient(url=QDRANT_URL, prefer_grpc=False) # Initialize Qdrant client.

         # Generate embeddings for the provided texts.
        text_embeddings = self.embeddings.embed_documents(texts)
        print("STARTING LOADING")

        for i in range(0, min(len(texts), len(metadata)), BATCH_SIZE):
            # Process embeddings in batches for efficient uploading.
            chunk_texts = texts[i:i + BATCH_SIZE]
            chunk_metadata = metadata[i:i + BATCH_SIZE]
            chunk_text_embeddings = text_embeddings[i:i + BATCH_SIZE]

            client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=i + n, # Unique ID for each point
                        vector=embedding, # Embedding vector.
                        payload={
                            'metadata': chunk_metadata[n], # Custom metadata for each text.
                            'page_content': chunk_texts[n] # Original text content.
                        }
                    )
                    for n, embedding in enumerate(chunk_text_embeddings) # Loop through embeddings to create points.
                ]
            )

            print(f"CHUNK [{i} - {i + BATCH_SIZE}] loaded")
            time.sleep(EMBEDDING_DELAY_SECONDS) # Add delay to avoid overloading Qdrant.

        client.close() # Close the client connection.
        print("Embeddings successfully loaded")

#####################################################################
# LLaMA Utility Functions

class LlamaUtility:
    """Utility class for managing LLaMA model operations."""
    def __init__(self):
        self.model, self.tokenizer = self.init_llama()

    def init_llama(self):
        """Initialize and return the LLaMA model and tokenizer."""
        torch.cuda.empty_cache() # Clear CUDA cache to free up memory.
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct" # Model identifier.
        tokenizer = AutoTokenizer.from_pretrained(model_id) # Load tokenizer.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, # Specify tensor type for model.
            device_map="auto", # Automatically place model on available device (CPU/GPU).
        )
        return model, tokenizer

    def build_prompt_messages(self, query: str, relevant_content: str) -> List[Dict]:
        """Build prompt messages to be used with the LLaMA model."""
        return [
            {
                "role": "system",
                "content": """You are a chat bot. Your goal is to analyze threat 
                intelligence reports. You're going to use relevant information from threat 
                intelligence reports and answer the user's question. Do not mention 
                in your response that I gave you extra context and if you can't find 
                something, just say you don't know the answer."""
            },
            {
                "role": "user",
                "content": f"Relevant Content From Reports: {relevant_content}\nUser question: {query}"
            },
        ]

    def prompt_llm(self, messages: List[Dict]) -> str:
        """Generate a response using the LLaMA language model based on the provided messages."""
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True, # Add a generation prompt to the input.
            return_tensors="pt" # Return PyTorch tensors.
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=500, # Maximum number of tokens to generate.
            eos_token_id=self.tokenizer.eos_token_id, # End-of-sequence token ID.
            do_sample=True, # Use sampling for generation.
            temperature=0.6, # Sampling temperature.
            top_p=0.9, # Top-p sampling threshold.
        )

        response = outputs[0][input_ids.shape[-1]:] # Extract the generated response.
        return self.tokenizer.decode(response, skip_special_tokens=True) # Decode and return the response.

#####################################################################
# PDF Ingestion Function

def ingest_pdf(file_name: str, collection_name: str = DEFAULT_COLLECTION_NAME, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
    """Ingest a PDF file, split its content into chunks, and convert it to embeddings for Qdrant."""
    try:
        loader = PyPDFLoader(file_name) # Load the PDF file.
    except Exception as e:
        print(f"Could not load {file_name}: {e}") # Print error if loading fails.
        return
    
    documents = loader.load() # Load documents from the PDF.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) # Initialize text splitter.
    texts = text_splitter.split_documents(documents) # Split documents into manageable chunks.

    qdrant_util = QdrantUtility(collection_name=collection_name) # Initialize Qdrant utility.
    qdrant_util.load_embeddings_with_custom_metadata(texts, [{}] * len(texts)) # Load embeddings into Qdrant.

    print("Qdrant Updated.") # Confirm update completion.

#####################################################################
# OTX Exploit Paper CSV Reader

def read_exploit_papers_csv_otx(file_path: str) -> Tuple[List[str], List[Dict]]:
    """Read and process the exploit papers CSV for OTX."""
    with open(file_path, "r") as file: 
        reader = csv.reader(file) # Initialize CSV reader.
        _ = next(reader) # Skip header row.

        texts = [] # List to store text contents.
        metadata = []  # List to store metadata.

        for row in reader:
            # Format each row into a text string and append to list.
            text_str = f"content: {row[4]} title: {row[5]} description: {row[6]}"
            texts.append(text_str)
            # Extract relevant metadata fields and append to list.
            metadata.append({
                'id': row[0],
                'indicator': row[1],
                'type': row[2],
                'created': row[3],
                'expiration': row[7],
                'is_active': row[8],
                'role': row[9],
                'pulse_id': row[10],
            })

        return texts, metadata # Return texts and metadata.
    
#####################################################################
# OTX_Rag Class

class OTX_Rag :
    """Class for combining Qdrant and LLaMA functionalities into a RAG system."""

    def __init__(self):
        self.qdrant_utils = QdrantUtility() # Initialize Qdrant utility.

    

    def build_messages(self, prompt: str, extra_context: str, relevent_content : str) -> list[str, str]:
        """Build messages for the chatbot based on provided prompt and context."""
        print(relevent_content) # Debug print statement.
        return [
            {
                'role': 'system',
                'content': f'''You are a chatbot. Answer the following question based on the data provided. Provided Data: {relevent_content}'''
            },
            {
                'role': 'user',
                'content': f'{prompt} - File Input: {extra_context}'
            }
        ]

    
    def get_messages_with_context(self, prompt: str, extra_context: str, num_chunks: int) -> tuple[list[dict[str, str]], list[str]]:
        """Retrieve relevant content from Qdrant and build messages for LLaMA."""

        relevant_context = self.qdrant_utils.retrieve_relevant_content(prompt, num_chunks)# Retrieve content from Qdrant.
        messages = self.build_messages(prompt, extra_context, relevant_context) # Build messages for the chatbot.
    
        return messages, relevant_context # Return messages and relevant content.


#####################################################################
# Main Application


if __name__ == '__main__':
    llama_util = LlamaUtility() # Initialize LLaMA utility.
    otx_rag = OTX_Rag() # Initialize OTX_Rag utility.

    MENU = """
    Rag Menu
    1) Prompt RAG
    2) Ingest PDF
    0) Exit
    > """

    while True:
        selection = input(MENU) # Display menu and get user selection.

        if selection == "1": # Prompt RAG option.
            query = input("User Question: ") # Get user question.
            print(f"DEBUG: Received user question: {query}") # Debug print statement.

            messages, relevant_content = otx_rag.get_messages_with_context(query, '', 5) # Get messages with context.
            print(f"DEBUG: Messages prepared for LLaMA: {messages}") # Debug print statement.

            response = llama_util.prompt_llm(messages) # Generate response using LLaMA.
            print("###############\n" + response + "\n#############\n")
            print("####### Relevant Context ########\n" + str(relevant_content) + "\n#############\n")

        elif selection == "2":  # Ingest PDF option.
            file_name = input("File Name: ") # Get PDF file name from user.
            print(f"DEBUG: Received file name for ingestion: {file_name}") # Debug print statement.
            ingest_pdf(file_name) # Ingest PDF data.

        else:
            print("Exiting...") # Exit message.
            break # Break out of the loop to exit the application.






















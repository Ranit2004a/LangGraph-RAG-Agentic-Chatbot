import os 
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import PINECONE_API_KEY

# set environ variable for pinecone
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY

# initialize pinecone client
pc=Pinecone(api_key=PINECONE_API_KEY)

#define embedding model
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

INDEX_NAME="rag-index"

#retriever function
def get_retriever():
    """Initializes and returns the Pinecone vector store retriever"""
    # ensure the index exists, create if not
    if INDEX_NAME not in pc.list_indexes().names():
        print("Creating new index")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print("Created pinecone index")

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return vectorstore.as_retriever()

def add_document(text_content: str):
    """Adds a document to the Pinecone vector store"""

    if not text_content:
        raise ValueError("Text content cannot be empty")

    # Create index if it doesn't exist
    if INDEX_NAME not in pc.list_indexes().names():
        print("Creating new index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Index created successfully")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    documents = text_splitter.create_documents([text_content])

    print("Splitting document into chunks for indexing...")

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    vectorstore.add_documents(documents)

    print("Document added to vector store")
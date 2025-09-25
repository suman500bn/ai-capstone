# This Python script contains the code for the second week of your AI project:
# a chatbot with Hybrid Search, Ensemble Retrieval, and Re-ranking.

# Make sure you have the necessary libraries installed.
# You will need to install this in your terminal/command prompt:
# pip install langchain-google-genai python-dotenv pypdf faiss-cpu sentence-transformers cohere rank-bm25

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
import cohere

# Load environment variables from a .env file
load_dotenv()

def load_documents(pdf_path):
    """
    Loads a PDF document and splits it into chunks.
    """
    try:
        # Load the document
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        faiss_vectorstore = FAISS.from_documents(texts, embeddings)

        return (faiss_vectorstore, texts)
        
    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found. Please ensure it's in the correct directory.")
        return []
    except Exception as e:
        print(f"An error occurred while loading documents: {e}")
        return []

def create_retriever(faiss_vectorstore, texts):
    """
    Loads a PDF, splits it into chunks, and creates a hybrid retriever
    with re-ranking using Cohere.
    """
    try:
        # Check for Cohere API Key
        cohere_api_key = os.environ.get("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable not set. Please add it to your .env file.")

        # 1. Create a semantic retriever (FAISS)
        faiss_retriever = faiss_vectorstore.as_retriever()
        print("Semantic (FAISS) retriever created.")

        # 2. Create a keyword retriever (BM25)
        bm25_retriever = BM25Retriever.from_documents(texts)
        print("Keyword (BM25) retriever created.")

        # 3. Combine them with EnsembleRetriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
        print("Ensemble retriever created.")

        # 4. Add a re-ranker for final relevance scoring
        compressor = CohereRerank(model="rerank-v3.5")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=ensemble_retriever
        )
        print("Re-ranking compressor added.")
        
        return compression_retriever

    except Exception as e:
        print(f"An error occurred during retriever creation: {e}")
        return None

def get_llm_response(question, retriever):
    """
    Uses a RetrievalQA chain to get a grounded response from the LLM,
    including the page numbers of the source documents.
    """
    try:
        # Get the API key from the loaded environment variables
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        # Initialize the LangChain model
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0
        )
        
        # Create a RetrievalQA chain, specifying return_source_documents=True
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True # This is the key change
        )
        
        # Invoke the chain to get the response and source documents
        result = qa_chain.invoke({"query": question})
        response_text = result['result']
        source_docs = result['source_documents']
        
        # Extract unique page numbers from the source documents
        page_numbers = set()
        for doc in source_docs:
            if 'page' in doc.metadata:
                page_numbers.add(doc.metadata['page'] + 1) # Page numbers are 0-indexed, so we add 1
        
        # Append the page numbers to the response
        if page_numbers:
            response_text += f"\n\n(Source Pages: {', '.join(map(str, sorted(list(page_numbers))))})"

        return response_text
        
    except Exception as e:
        return f"An error occurred: {e}"

# This is the main loop for the Week 2 chatbot.
if __name__ == "__main__":
    pdf_file_path = os.path.join("data", "Johnson_H.pdf")
    
    print("--- Week 2: Chatbot with Hybrid Search & Re-ranking ---")
    print(f"I'm now an expert on the document: {pdf_file_path}")
    print("Ask me questions about risk management. Type 'exit' to quit.")

    # Create the retriever from the PDF
    vectorstore, texts = load_documents(pdf_file_path)
    retriever = create_retriever(vectorstore, texts)

    if retriever:
        while True:
            user_query = input("You: ")
            if user_query.lower() == 'exit':
                break
            
            # Get the chatbot's response
            bot_response = get_llm_response(user_query, retriever)
            print("Bot:", bot_response)

    print("---------------------------------")

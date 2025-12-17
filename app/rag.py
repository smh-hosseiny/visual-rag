import os
import shutil
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

class RAGManager:
    def __init__(self):
        print("🔄 Loading Local Embedding Model (all-MiniLM-L6-v2)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'} 
        )
        self.persist_dir = "./chroma_db"
        self._init_db()
        print("✅ Local Embeddings Ready.")

    def _init_db(self):
        """Initialize the ChromaDB connection"""
        self.vector_store = Chroma(
            collection_name="market_research",
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )


    def clear(self):
        """⚠️ Wipes the entire vector database for a fresh start"""
        print("🧹 Clearing Vector Database...")
        # 1. Delete the in-memory object
        self.vector_store = None
        
        # 2. Nuke the folder from disk
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            
        # 3. Re-initialize
        self._init_db()
        print("✨ Database wiped and ready for new task.")
        

    def add_documents(self, text_content: str, source: str):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = [Document(page_content=x, metadata={"source": source}) 
                for x in text_splitter.split_text(text_content)]
        self.vector_store.add_documents(docs)

    def retrieve(self, query: str, k: int = 20):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)
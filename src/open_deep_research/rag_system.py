"""
RAG (Retrieval Augmented Generation) system for semantic file search.
"""

import asyncio
import glob
import hashlib
import pickle
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration


class RAGFileSearchSystem:
    """RAG system for semantic search across multiple files."""
    
    def __init__(self, cache_dir: str = "./.rag_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.vectorstore = None
        self.file_hashes = {}
        self.last_update = None
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate hash of file for caching."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_cache_key(self, file_paths: List[str]) -> str:
        """Generate cache key from file paths and their hashes."""
        hashes = []
        for fp in file_paths:
            try:
                hashes.append(self._get_file_hash(fp))
            except:
                hashes.append("missing")
        return hashlib.md5("|".join(sorted(hashes)).encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> bool:
        """Try to load vectorstore from cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.vectorstore = data['vectorstore']
                    self.file_hashes = data['file_hashes']
                    self.last_update = data['timestamp']
                print(f"✅ Loaded RAG index from cache: {cache_key}")
                return True
            except Exception as e:
                print(f"⚠️ Cache load failed: {e}")
        return False
    
    def _save_to_cache(self, cache_key: str):
        """Save vectorstore to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            data = {
                'vectorstore': self.vectorstore,
                'file_hashes': self.file_hashes,
                'timestamp': datetime.now()
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Saved RAG index to cache: {cache_key}")
        except Exception as e:
            print(f"⚠️ Cache save failed: {e}")
    
    def _needs_rebuild(self, file_paths: List[str]) -> bool:
        """Check if RAG index needs to be rebuilt."""
        if not self.vectorstore:
            return True
        
        # Check if files changed
        current_hashes = {}
        for fp in file_paths:
            try:
                current_hashes[fp] = self._get_file_hash(fp)
            except:
                current_hashes[fp] = "missing"
        
        return current_hashes != self.file_hashes
    
    def _build_rag_index(self, file_paths: List[str]):
        """Build FAISS vector index from files."""
        all_documents = []
        
        print(f"🔨 Building RAG index for {len(file_paths)} files...")
        
        for file_path in file_paths:
            path = Path(file_path)
            
            if not path.exists():
                print(f"⚠️ File not found: {file_path}")
                continue
            
            try:
                # Calculate file hash for caching
                file_hash = self._get_file_hash(file_path)
                self.file_hashes[file_path] = file_hash
                
                # Load file
                loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata.update({
                        "source": str(file_path),
                        "filename": path.name,
                        "file_size": path.stat().st_size,
                        "modified": path.stat().st_mtime
                    })
                
                all_documents.extend(docs)
                print(f"📄 Loaded {len(docs)} chunks from {path.name}")
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        
        if not all_documents:
            raise ValueError("No valid documents found!")
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(all_documents)
        print(f"📊 Split into {len(chunks)} chunks")
        
        # Create embeddings
        print("🧠 Creating embeddings...")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            chunk_size=100  # Process in batches
        )
        
        # Build vectorstore
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        print(f"✅ Built RAG index with {len(chunks)} vectors")
        
        # Update timestamp
        self.last_update = datetime.now()
    
    async def search(
        self, 
        query: str, 
        file_paths: List[str], 
        k: int = 5,
        config: Optional[RunnableConfig] = None
    ) -> str:
        """Perform semantic search across files."""
        
        # Get API key from config if needed
        configurable = Configuration.from_runnable_config(config)
        
        # Check cache and rebuild if needed
        cache_key = self._get_cache_key(file_paths)
        
        if not self._load_from_cache(cache_key) or self._needs_rebuild(file_paths):
            self._build_rag_index(file_paths)
            self._save_to_cache(cache_key)
        
        # Perform semantic search
        print(f"🔍 Semantic search: '{query}' (k={k})")
        
        try:
            # Get similar documents
            docs = self.vectorstore.similarity_search(query, k=k)
            
            if not docs:
                return "❌ No relevant information found in files."
            
            # Format results
            results = []
            results.append(f"## 🧠 Semantic Search Results")
            results.append(f"**Query:** '{query}'")
            results.append(f"**Files searched:** {len(file_paths)}")
            results.append(f"**Most relevant passages:**\n")
            
            for i, doc in enumerate(docs, 1):
                # Extract metadata
                source = doc.metadata.get('filename', 'Unknown')
                score = doc.metadata.get('score', 'N/A')
                
                # Highlight query terms in context
                content = doc.page_content
                
                # Format result
                result = f"""
### 📄 Result {i}: {source}
**Relevance score:** {score if score != 'N/A' else 'High'}
**Source file:** `{doc.metadata.get('source', 'Unknown')}`

**Content:**
{content[:800]}{'...' if len(content) > 800 else ''}

---
"""
                results.append(result)
            
            # Add search statistics
            results.append(f"\n**Search completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            results.append(f"**Index last updated:** {self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else 'Never'}")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"❌ Search error: {str(e)}"
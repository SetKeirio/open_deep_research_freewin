"""
RAG Manager - централизованное управление RAG системами.
"""

import asyncio
import hashlib
import pickle
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration


class RAGManager:
    """Singleton менеджер для RAG систем."""
    
    _instance = None
    _rag_systems: Dict[str, 'RAGSystem'] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_rag_system(self, config_key: str) -> 'RAGSystem':
        """Получить или создать RAG систему по ключу конфигурации."""
        if config_key not in self._rag_systems:
            self._rag_systems[config_key] = RAGSystem()
        return self._rag_systems[config_key]
    
    async def search(
        self,
        query: str,
        file_paths: List[str],
        k: int = 5,
        config: Optional[RunnableConfig] = None
    ) -> str:
        """Выполнить поиск через RAG систему."""
        configurable = Configuration.from_runnable_config(config)
        
        # Создаем уникальный ключ для конфигурации
        config_hash = hashlib.md5(
            f"{sorted(file_paths)}".encode()
        ).hexdigest()
        
        rag_system = self.get_rag_system(config_hash)
        return await rag_system.search(query, file_paths, k, config)


class RAGSystem:
    """Основная RAG система."""
    
    def __init__(self, cache_dir: str = "./.rag_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.vectorstore = None
        self.file_hashes = {}
        self.last_update = None
    
    def _get_file_hash(self, file_path: str) -> str:
        """Вычислить хэш файла для кэширования."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return "missing"
    
    def _get_cache_key(self, file_paths: List[str]) -> str:
        """Сгенерировать ключ кэша."""
        hashes = [self._get_file_hash(fp) for fp in file_paths]
        return hashlib.md5("|".join(sorted(hashes)).encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> bool:
        """Загрузить из кэша."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.vectorstore = data['vectorstore']
                    self.file_hashes = data['file_hashes']
                    self.last_update = data['timestamp']
                print(f"✅ Загружен RAG индекс из кэша: {cache_key}")
                return True
            except Exception as e:
                print(f"⚠️ Ошибка загрузки кэша: {e}")
        return False
    
    def _save_to_cache(self, cache_key: str):
        """Сохранить в кэш."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            data = {
                'vectorstore': self.vectorstore,
                'file_hashes': self.file_hashes,
                'timestamp': datetime.now()
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Сохранен RAG индекс в кэш: {cache_key}")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения кэша: {e}")
    
    def _needs_rebuild(self, file_paths: List[str]) -> bool:
        """Проверить, нужно ли перестроить индекс."""
        if not self.vectorstore:
            return True
        
        # Проверить, изменились ли файлы
        current_hashes = {fp: self._get_file_hash(fp) for fp in file_paths}
        return current_hashes != self.file_hashes
    
    def _build_rag_index(self, file_paths: List[str]):
        """Построить векторный индекс из файлов."""
        all_documents = []
        
        print(f"🔨 Строим RAG индекс для {len(file_paths)} файлов...")
        
        for file_path in file_paths:
            path = Path(file_path)
            
            if not path.exists():
                print(f"⚠️ Файл не найден: {file_path}")
                continue
            
            try:
                # Вычислить хэш файла
                file_hash = self._get_file_hash(file_path)
                self.file_hashes[file_path] = file_hash
                
                # ЗАГРУЗКА ФАЙЛА - исправленная версия
                print(f"📖 Читаем файл: {path.name}")
                
                # Читаем файл вручную для отладки
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                print(f"📏 Размер файла: {len(content)} символов")
                print(f"📝 Первые 500 символов: {content[:500]}...")
                
                # Создаем документы вручную
                docs = [Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "filename": path.name,
                        "file_size": len(content)
                    }
                )]
                
                all_documents.extend(docs)
                print(f"📄 Создано {len(docs)} документ из {path.name}")
                
            except Exception as e:
                print(f"❌ Ошибка загрузки {file_path}: {e}")
                import traceback
                traceback.print_exc()
        
        if not all_documents:
            raise ValueError("Не найдено валидных документов!")
        
        # Разбить на чанки
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
        chunks = splitter.split_documents(all_documents)
        print(f"📊 Разбито на {len(chunks)} чанков")
        for i, chunk in enumerate(chunks[:3]):  # Показать первые 3 чанка
            print(f"  Чанк {i+1}: {len(chunk.page_content)} символов")
            print(f"  Содержимое: {chunk.page_content[:200]}...")
        
        # Создать эмбеддинги
        print("🧠 Создаем эмбеддинги...")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            chunk_size=100
        )
        
        # Построить векторное хранилище
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        print(f"✅ Построен RAG индекс с {len(chunks)} векторами")
        
        # Обновить время
        self.last_update = datetime.now()
    
    async def search(
        self, 
        query: str, 
        file_paths: List[str], 
        k: int = 5,
        config: Optional[RunnableConfig] = None
    ) -> str:
        """Выполнить семантический поиск."""
        
        # Проверить кэш и перестроить если нужно
        cache_key = self._get_cache_key(file_paths)
        
        if not self._load_from_cache(cache_key) or self._needs_rebuild(file_paths):
            self._build_rag_index(file_paths)
            self._save_to_cache(cache_key)
        
        # Выполнить семантический поиск
        print(f"🔍 Семантический поиск: '{query}' (k={k})")
        
        try:
            # Получить похожие документы
            docs = self.vectorstore.similarity_search(query, k=k)
            
            if not docs:
                return "❌ Не найдено релевантной информации в файлах."
            
            # Форматировать результаты
            results = []
            results.append(f"## 🧠 Результаты семантического поиска")
            results.append(f"**Запрос:** '{query}'")
            results.append(f"**Файлов поискано:** {len(file_paths)}")
            results.append(f"**Самые релевантные отрывки:**\n")
            
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('filename', 'Неизвестно')
                score = doc.metadata.get('score', 'Н/Д')
                
                result = f"""
### 📄 Результат {i}: {source}
**Оценка релевантности:** {score if score != 'Н/Д' else 'Высокая'}
**Файл источника:** `{doc.metadata.get('source', 'Неизвестно')}`

**Содержание:**
{doc.page_content[:800]}{'...' if len(doc.page_content) > 800 else ''}

---
"""
                results.append(result)
            
            # Добавить статистику поиска
            results.append(f"\n**Поиск завершен:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"❌ Ошибка поиска: {str(e)}"


# Глобальный экземпляр менеджера
_rag_manager_instance = None

def get_rag_manager() -> RAGManager:
    """Получить глобальный экземпляр менеджера RAG."""
    global _rag_manager_instance
    if _rag_manager_instance is None:
        _rag_manager_instance = RAGManager()
    return _rag_manager_instance
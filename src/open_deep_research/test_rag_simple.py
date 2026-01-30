"""
Простой тест RAG системы без запуска всего агента.
"""

import asyncio
import os
import sys

# Добавляем путь к проекту
sys.path.append('.')

async def test_rag_direct():
    """Прямой тест RAG системы."""
    print("🧪 Тестируем RAG систему напрямую...")
    
    try:
        # Импортируем менеджер
        from open_deep_research.rag_manager import get_rag_manager
        
        # Получаем менеджер
        rag_manager = get_rag_manager()
        print("✅ RAG менеджер загружен")
        
        # Тестовые файлы - ИСПРАВЛЕННЫЕ ПУТИ!
        # Ищем файлы на уровень выше (в src/)
        import os
        from pathlib import Path
        
        current_dir = Path(__file__).parent  # src/open_deep_research
        parent_dir = current_dir.parent      # src/
        
        file_paths = [
            str(parent_dir / "data.txt"),
            str(parent_dir / "link_fraudulent.txt")
        ]
        
        # Проверяем файлы
        for fp in file_paths:
            path = Path(fp)
            if path.exists():
                print(f"✅ Файл существует: {fp}")
                # Читаем для проверки
                with open(fp, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"  📏 Размер: {len(content)} символов")
                    print(f"  📝 Начало: {content[:100]}...")
            else:
                print(f"❌ Файл отсутствует: {fp}")
                print(f"  📁 Текущая директория: {os.getcwd()}")
                print(f"  📁 Родительская директория: {parent_dir}")
                print(f"  📁 Содержимое родительской директории:")
                for f in parent_dir.glob('*.txt'):
                    print(f"    - {f.name}")
                # НЕ создаем тестовый файл - используем реальные!
                return False
        
        # Тестовый запрос
        query = "лучшие компании обувь 2025"
        print(f"\n🔍 Запрос: '{query}'")
        
        # Выполняем поиск
        result = await rag_manager.search(
            query=query,
            file_paths=file_paths,
            k=3,
            config=None
        )
        
        print("\n📊 Результат поиска:")
        print("=" * 60)
        print(result[:1500] + "..." if len(result) > 1500 else result)
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_rag_tool():
    try:
        from open_deep_research.utils import rag_file_search
        from pathlib import Path
        
        # Определяем правильные пути
        current_dir = Path(__file__).parent  # src/open_deep_research
        parent_dir = current_dir.parent      # src/
        
        config = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            # Явно задайте правильные пути
            "local_file_path": [str(parent_dir / "data.txt"), str(parent_dir / "link_fraudulent.txt")]
        }
        query = "footwear manufacturing 2025"
        
        # ИСПРАВЛЕНО: rag_file_search - это функция, а не инструмент!
        # Используйте вызов функции напрямую
        result = await rag_file_search(
            query=query, 
            config=config,
            k=5
        )
        
        print(f"🔍 Запрос инструменту: '{query}'")
        print(f"📝 Результат инструмента: {result}")
        return True
    except Exception as e:
        print(f"❌ Ошибка инструмента: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Запуск тестов RAG системы")
    print("=" * 60)
    
    # Тест 1: Прямой вызов менеджера
    asyncio.run(test_rag_direct())
    
    # Тест 2: Через инструмент
    asyncio.run(test_rag_tool())
    
    print("\n✅ Тесты завершены")
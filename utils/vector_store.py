"""
Vector Store Module
ChromaDB ile embedding'leri saklar ve benzerlik araması yapar
"""

import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    """
    ChromaDB kullanarak embedding'leri saklar ve arama yapar
    """
    
    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = None):
        """
        Args:
            collection_name: ChromaDB koleksiyon adı
            persist_directory: Veritabanı klasörü
        """
        if persist_directory is None:
            persist_directory = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        
        # ChromaDB klasörünü oluştur
        os.makedirs(persist_directory, exist_ok=True)
        
        # ChromaDB client oluştur
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection_name = collection_name
        
        # Koleksiyonu al veya oluştur
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✅ Koleksiyon yüklendi: {collection_name}")
            print(f"📊 Mevcut döküman sayısı: {self.collection.count()}")
        except:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"✅ Yeni koleksiyon oluşturuldu: {collection_name}")
    
    def add_chunks(self, chunks_with_embeddings: List[Dict]) -> None:
        """
        Chunk'ları ve embeddings'leri ChromaDB'ye ekle
        
        Args:
            chunks_with_embeddings: [{'text': str, 'embedding': List[float], 'metadata': dict}, ...]
        """
        if not chunks_with_embeddings:
            print("⚠️  Eklenecek chunk yok!")
            return
        
        # Verileri hazırla
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks_with_embeddings):
            # Benzersiz ID oluştur
            chunk_id = f"chunk_{self.collection.count() + i}"
            ids.append(chunk_id)
            
            # Embedding ve metin
            embeddings.append(chunk['embedding'])
            documents.append(chunk['text'])
            
            # Metadata'yı temizle (ChromaDB sadece str, int, float, bool kabul eder)
            metadata = {}
            for key, value in chunk['metadata'].items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                elif isinstance(value, list):
                    metadata[key] = str(value)  # Listeyi string'e çevir
                else:
                    metadata[key] = str(value)
            
            metadatas.append(metadata)
        
        # ChromaDB'ye ekle
        print(f"⏳ {len(chunks_with_embeddings)} chunk ChromaDB'ye ekleniyor...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"✅ Ekleme tamamlandı! Toplam döküman: {self.collection.count()}")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Query embedding'ine benzer chunk'ları bul
        
        Args:
            query_embedding: Sorgu vektörü
            top_k: Kaç sonuç döndürsün
            
        Returns:
            List[Dict]: [{'text': str, 'metadata': dict, 'distance': float}, ...]
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Sonuçları formatla
        formatted_results = []
        
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'id': results['ids'][0][i]
                })
        
        return formatted_results
    
    def delete_collection(self) -> None:
        """Koleksiyonu tamamen sil"""
        self.client.delete_collection(name=self.collection_name)
        print(f"🗑️  Koleksiyon silindi: {self.collection_name}")
    
    def get_stats(self) -> Dict:
        """Veritabanı istatistikleri"""
        return {
            'collection_name': self.collection_name,
            'total_documents': self.collection.count()
        }


# Test fonksiyonu
if __name__ == "__main__":
    print("\n" + "="*60)
    print("CHROMADB VECTOR STORE TEST")
    print("="*60 + "\n")
    
    try:
        # Vector store oluştur
        vector_store = VectorStore(collection_name="test_collection")
        
        # Test verileri (manuel embedding'ler)
        test_chunks = [
            {
                'text': 'Python bir programlama dilidir',
                'embedding': [0.1, 0.2, 0.3, 0.4, 0.5],
                'metadata': {'source': 'test', 'chunk_id': 0}
            },
            {
                'text': 'JavaScript web geliştirmede kullanılır',
                'embedding': [0.2, 0.3, 0.4, 0.5, 0.6],
                'metadata': {'source': 'test', 'chunk_id': 1}
            },
            {
                'text': 'Makine öğrenimi yapay zeka dalıdır',
                'embedding': [0.9, 0.8, 0.7, 0.6, 0.5],
                'metadata': {'source': 'test', 'chunk_id': 2}
            }
        ]
        
        # Verileri ekle
        print("📝 Test verileri ekleniyor...\n")
        vector_store.add_chunks(test_chunks)
        
        # Arama testi
        print("\n" + "="*60)
        print("ARAMA TESTİ")
        print("="*60 + "\n")
        
        query_embedding = [0.15, 0.25, 0.35, 0.45, 0.55]  # Python'a yakın
        print(f"🔍 Sorgu: [0.15, 0.25, 0.35, 0.45, 0.55] (Python'a benzer)")
        
        results = vector_store.search(query_embedding, top_k=3)
        
        print(f"\n📊 Bulunan {len(results)} sonuç:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['text']}")
            print(f"   Benzerlik skoru: {result['distance']:.4f}")
            print(f"   Metadata: {result['metadata']}\n")
        
        # İstatistikler
        stats = vector_store.get_stats()
        print("="*60)
        print(f"📈 İstatistikler:")
        print(f"   Koleksiyon: {stats['collection_name']}")
        print(f"   Toplam döküman: {stats['total_documents']}")
        
        # Temizlik
        print("\n🗑️  Test koleksiyonu siliniyor...")
        vector_store.delete_collection()
        print("✅ Test tamamlandı!")
        
    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()
        
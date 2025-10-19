"""
Embeddings Module
Metinleri vektörlere (embeddings) çevirir - OpenAI ile
"""

from typing import List
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# .env dosyasını yükle
load_dotenv()


class EmbeddingGenerator:
    """
    OpenAI ile metinleri embedding vektörlerine çevirir
    """
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Args:
            model_name: OpenAI embedding model ismi
                       Seçenekler:
                       - text-embedding-3-small (512 dim, ucuz)
                       - text-embedding-3-large (3072 dim, daha iyi)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "❌ OPENAI_API_KEY bulunamadı!\n"
                "Çözüm: .env dosyasına ekleyin:\n"
                "OPENAI_API_KEY=sk-proj-xxxxxxxxx"
            )
        
        self.model_name = model_name
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key
        )
        
        print(f"✅ OpenAI Embedding modeli yüklendi: {model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Tek bir metni embedding'e çevir
        
        Args:
            text: Metin
            
        Returns:
            List[float]: Embedding vektörü
        """
        return self.embeddings.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Birden fazla metni embedding'e çevir
        
        Args:
            texts: Metin listesi
            
        Returns:
            List[List[float]]: Embedding vektörleri
        """
        return self.embeddings.embed_documents(texts)
    
    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        Chunk'ları embedding'e çevir
        
        Args:
            chunks: [{'text': str, 'metadata': dict}, ...]
            
        Returns:
            List[dict]: [{'text': str, 'embedding': List[float], 'metadata': dict}, ...]
        """
        # Metinleri çıkar
        texts = [chunk['text'] for chunk in chunks]
        
        # Embeddings oluştur
        print(f"⏳ {len(texts)} chunk için embedding oluşturuluyor...")
        embeddings = self.embed_texts(texts)
        print(f"✅ Embeddings hazır!")
        
        # Chunk'lara ekle
        result = []
        for chunk, embedding in zip(chunks, embeddings):
            result.append({
                'text': chunk['text'],
                'embedding': embedding,
                'metadata': chunk['metadata']
            })
        
        return result
    
    def get_embedding_dimension(self) -> int:
        """Embedding vektörünün boyutunu döndür"""
        test_embedding = self.embed_text("test")
        return len(test_embedding)


# Test fonksiyonu
if __name__ == "__main__":
    print("\n" + "="*60)
    print("OPENAI EMBEDDING GENERATOR TEST")
    print("="*60 + "\n")
    
    try:
        # Embedding generator oluştur
        generator = EmbeddingGenerator()
        
        # Test metinleri
        test_texts = [
            "Yapay zeka çok gelişti",
            "AI teknolojisi ilerliyor",
            "Bugün hava çok güzel"
        ]
        
        print(f"📝 Test metinleri:")
        for i, text in enumerate(test_texts, 1):
            print(f"  {i}. {text}")
        
        # Embeddings oluştur
        print(f"\n⏳ Embeddings oluşturuluyor...")
        embeddings = generator.embed_texts(test_texts)
        
        print(f"\n✅ {len(embeddings)} embedding oluşturuldu!")
        print(f"📊 Embedding boyutu: {len(embeddings[0])} boyutlu vektör")
        print(f"\n🔢 İlk embedding'in ilk 10 değeri:")
        print(f"   {[round(x, 4) for x in embeddings[0][:10]]}")
        
        # Benzerlik analizi
        import numpy as np
        
        def cosine_similarity(vec1, vec2):
            """İki vektör arasındaki kosinüs benzerliği"""
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
        sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])
        
        print("\n" + "="*60)
        print("BENZERLİK ANALİZİ (Cosine Similarity)")
        print("="*60)
        print(f"\n'Yapay zeka' ↔ 'AI teknolojisi':  {sim_1_2:.4f} (Yüksek olmalı)")
        print(f"'Yapay zeka' ↔ 'Hava durumu':     {sim_1_3:.4f} (Düşük olmalı)")
        
        if sim_1_2 > sim_1_3:
            print("\n✅ Başarılı! Benzer anlamlı metinler daha yakın vektörlere sahip.")
        else:
            print("\n⚠️  Beklenmedik sonuç!")
        
    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        print("\n💡 Çözüm:")
        print("1. .env dosyasını aç")
        print("2. Şu satırı ekle:")
        print("   OPENAI_API_KEY=sk-proj-xxxxxxxxx")
        print("3. Dosyayı kaydet")
        print("4. Tekrar dene!")
        
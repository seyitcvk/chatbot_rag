"""
Chunking Module
Metni küçük, yönetilebilir parçalara böler
"""

from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextChunker:
    """
    Metni chunklara (parçalara) böler
    
    Chunking Stratejisi:
    - Recursive: Paragraf ve cümle sınırlarına göre böler
    - Overlap: Parçalar arasında örtüşme bırakır (context korunur)
    """
    
    def __init__(
        self, 
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        """
        Args:
            chunk_size: Her chunk'ın maksimum karakter sayısı
            chunk_overlap: Chunklar arası örtüşme miktarı
            separators: Bölme için kullanılacak ayraçlar
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Varsayılan ayraçlar (öncelik sırasına göre)
        if separators is None:
            separators = [
                "\n\n",  # Paragraf
                "\n",    # Satır
                ". ",    # Cümle
                "! ",    # Ünlem
                "? ",    # Soru
                ", ",    # Virgül
                " ",     # Boşluk
                ""       # Karakter
            ]
        
        # LangChain'in RecursiveCharacterTextSplitter'ını kullan
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Metni chunklara böler
        
        Args:
            text: Bölünecek metin
            metadata: Chunk'lara eklenecek metadata
            
        Returns:
            List[Dict]: Her chunk için {'text': str, 'metadata': dict}
        """
        if not text or not text.strip():
            return []
        
        # Metni böl
        chunks = self.splitter.split_text(text)
        
        # Her chunk için dict oluştur
        result = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                'text': chunk,
                'metadata': {
                    'chunk_id': i,
                    'chunk_size': len(chunk),
                    'total_chunks': len(chunks),
                    **(metadata or {})
                }
            }
            result.append(chunk_data)
        
        return result
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Birden fazla dokümanı chunklar
        
        Args:
            documents: [{'content': str, 'metadata': dict}, ...]
            
        Returns:
            List[Dict]: Tüm chunklar
        """
        all_chunks = []
        
        for doc in documents:
            content = doc.get('content', '')
            doc_metadata = doc.get('metadata', {})
            
            # Dokümanı chunkla
            chunks = self.chunk_text(content, doc_metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict:
        """
        Chunking istatistikleri
        
        Args:
            chunks: Chunk listesi
            
        Returns:
            Dict: İstatistikler
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0
            }
        
        chunk_sizes = [len(c['text']) for c in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes)
        }


# Test fonksiyonu
if __name__ == "__main__":
    # Test metni
    test_text = """
    Yapay zeka, bilgisayar sistemlerinin insan zekasını taklit etme yeteneğidir.
    Bu teknoloji, son yıllarda büyük gelişmeler kaydetmiştir.
    
    Makine öğrenimi, yapay zekanın bir alt dalıdır. Veri analizinde kullanılır.
    Derin öğrenme ise makine öğreniminin en gelişmiş halidir.
    
    Doğal dil işleme, bilgisayarların insan dilini anlamasını sağlar.
    ChatGPT gibi modeller bu teknolojiye dayanır.
    """
    
    # Chunker oluştur
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    # Metni chunkla
    chunks = chunker.chunk_text(test_text, metadata={'source': 'test'})
    
    # Sonuçları göster
    print(f"\n{'='*60}")
    print(f"CHUNKING TEST SONUÇLARI")
    print(f"{'='*60}\n")
    
    print(f"Toplam Chunk Sayısı: {len(chunks)}\n")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(f"Boyut: {chunk['metadata']['chunk_size']} karakter")
        print(f"İçerik: {chunk['text'][:80]}...")
        print()
    
    # İstatistikler
    stats = chunker.get_chunk_statistics(chunks)
    print(f"\n{'='*60}")
    print(f"İSTATİSTİKLER")
    print(f"{'='*60}")
    for key, value in stats.items():
        print(f"{key}: {value}")
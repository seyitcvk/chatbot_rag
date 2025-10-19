"""
RAG Chat Application
Streamlit ile PDF/CSV yükleyip soru-cevap yapabileceğiniz arayüz
"""

import streamlit as st
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Kendi modüllerimizi import et
from utils.document_loader import DocumentLoader
from utils.chunking import TextChunker
from utils.embeddings import EmbeddingGenerator
from utils.vector_store import VectorStore

# .env dosyasını yükle
load_dotenv()

# Sayfa ayarları
st.set_page_config(
    page_title="Akıllı Okuyucu - RAG Chat",
    page_icon="📖",
    layout="wide"
)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Session state'leri initialize et
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'embedding_generator' not in st.session_state:
    st.session_state.embedding_generator = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False


def process_uploaded_file(uploaded_file):
    """Yüklenen dosyayı işle"""
    
    # Geçici olarak kaydet
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)


def initialize_rag_system():
    """RAG sistemini başlat"""
    if st.session_state.embedding_generator is None:
        with st.spinner("🔧 Embedding modeli yükleniyor..."):
            st.session_state.embedding_generator = EmbeddingGenerator()
    
    if st.session_state.vector_store is None:
        with st.spinner("🔧 Vector store başlatılıyor..."):
            st.session_state.vector_store = VectorStore(collection_name="rag_documents")


def process_documents(file_paths):
    """Dokümanları işle ve ChromaDB'ye ekle"""
    
    # 1. Dokümanları yükle
    with st.spinner("📄 Dokümanlar okunuyor..."):
        loader = DocumentLoader()
        documents = []
        for file_path in file_paths:
            try:
                doc = loader.load_document(file_path)
                documents.append(doc)
                st.success(f"✅ {Path(file_path).name} okundu!")
            except Exception as e:
                st.error(f"❌ {Path(file_path).name} okunamadı: {str(e)}")
    
    if not documents:
        st.error("Hiç doküman yüklenemedi!")
        return False
    
    # 2. Chunking
    with st.spinner("✂️ Metinler bölünüyor..."):
        chunker = TextChunker(
            chunk_size=int(os.getenv("CHUNK_SIZE", 500)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50))
        )
        chunks = chunker.chunk_documents(documents)
        st.info(f"📊 {len(chunks)} chunk oluşturuldu!")
    
    # 3. Embeddings oluştur
    with st.spinner("🧮 Embeddings oluşturuluyor..."):
        chunks_with_embeddings = st.session_state.embedding_generator.embed_chunks(chunks)
        st.success(f"✅ {len(chunks_with_embeddings)} embedding hazır!")
    
    # 4. ChromaDB'ye ekle
    with st.spinner("💾 ChromaDB'ye kaydediliyor..."):
        st.session_state.vector_store.add_chunks(chunks_with_embeddings)
        st.success("✅ Dokümanlar veritabanına eklendi!")
    
    return True


def get_rag_response(user_query):
    """RAG ile cevap üret"""
    
    # 1. Query'yi embedding'e çevir
    query_embedding = st.session_state.embedding_generator.embed_text(user_query)
    
    # 2. Benzer chunk'ları bul
    results = st.session_state.vector_store.search(query_embedding, top_k=5)
    
    # Eğer hiç benzer chunk yoksa
    if not results:
        return """Üzgünüm, yüklediğiniz dokümanlarda bu soruyla ilgili bilgi bulamadım. 📚
        
Ben **Akıllı Okuyucu**'yum ve sadece yüklediğiniz PDF veya CSV dosyalarındaki bilgileri kullanarak sorularınızı cevaplayabilirim.

Lütfen dokümanlarınızla ilgili sorular sorun! 😊""", []
    
    # 3. Context oluştur
    context = "\n\n".join([r['text'] for r in results])
    
    # 4. OpenAI ile cevap üret
    prompt = f"""Sen Akıllı Okuyucu'sun. Sadece verilen context bilgisini kullanarak soruyu cevapla.

ÖNEMLİ KURALLAR:
1. Eğer context'te sorunun cevabı YOKSA, şunu söyle: "Bu bilgi yüklediğiniz dokümanlarda bulunmuyor. Ben sadece yüklediğiniz dokümanlar hakkında bilgi verebiliyorum."
2. ASLA context dışında bilgi kullanma!
3. Eğer soru dokümanlarla alakasızsa (örn: hava durumu, yemek tarifi, genel bilgi), şunu söyle: "Bu soru yüklediğiniz dokümanlarla ilgili değil. Ben Akıllı Okuyucu'yum ve sadece yüklediğiniz dosyalardaki bilgileri kullanarak cevap verebilirim."

Context:
{context}

Soru: {user_query}

Cevap:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Sen Akıllı Okuyucu'sun. SADECE verilen context'teki bilgileri kullan. Context dışında ASLA bilgi verme!"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    
    return answer, results


# --- STREAMLIT UI ---

st.title("📖 Akıllı Okuyucu - RAG Chat Uygulaması")
st.markdown("PDF veya CSV yükleyin ve dokümanlarınızla sohbet edin!")

# Sidebar - Dosya Yükleme
with st.sidebar:
    st.header("📁 Doküman Yükleme")
    
    uploaded_files = st.file_uploader(
        "PDF veya CSV dosyası yükleyin",
        type=['pdf', 'csv'],
        accept_multiple_files=True
    )
    
    if st.button("🚀 Dokümanları İşle", type="primary"):
        if not uploaded_files:
            st.error("Lütfen önce dosya yükleyin!")
        else:
            # RAG sistemini başlat
            initialize_rag_system()
            
            # Dosyaları kaydet
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = process_uploaded_file(uploaded_file)
                file_paths.append(file_path)
            
            # İşle
            if process_documents(file_paths):
                st.session_state.documents_loaded = True
                st.balloons()
    
    # İstatistikler
    if st.session_state.vector_store:
        st.divider()
        st.subheader("📊 İstatistikler")
        stats = st.session_state.vector_store.get_stats()
        st.metric("Toplam Chunk", stats['total_documents'])
    
    # Veritabanını sıfırla
    st.divider()
    if st.button("🗑️ Veritabanını Sıfırla", type="secondary"):
        if st.session_state.vector_store:
            st.session_state.vector_store.delete_collection()
            st.session_state.vector_store = None
            st.session_state.documents_loaded = False
            st.session_state.chat_history = []
            st.success("Veritabanı sıfırlandı!")
            st.rerun()


# Ana alan - Chat
if not st.session_state.documents_loaded:
    st.info("👈 Sol taraftan doküman yükleyip işleyin, ardından soru sorabilirsiniz!")
else:
    st.success("✅ Dokümanlar yüklendi! Soru sorabilirsiniz.")
    
    # Chat history göster
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Eğer kaynak varsa göster
            if "sources" in message and message["sources"]:
                with st.expander("📚 Kaynaklar"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Kaynak {i}** (Benzerlik: {source['distance']:.4f})")
                        st.text(source['text'][:200] + "...")
                        st.json(source['metadata'])
    
    # Chat input
    if prompt := st.chat_input("Sorunuzu yazın..."):
        # Kullanıcı mesajını göster
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # RAG cevabı al
        with st.chat_message("assistant"):
            with st.spinner("🤔 Düşünüyorum..."):
                answer, sources = get_rag_response(prompt)
                st.markdown(answer)
                
                # Kaynakları göster
                if sources:
                    with st.expander("📚 Kaynaklar"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Kaynak {i}** (Benzerlik: {source['distance']:.4f})")
                            st.text(source['text'][:200] + "...")
                            st.json(source['metadata'])
        
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": answer,
            "sources": sources
        })
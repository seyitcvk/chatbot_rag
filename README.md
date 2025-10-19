# 📖 Akıllı Okuyucu - RAG Chat Uygulaması

PDF veya CSV dosyalarınızı yükleyin ve dokümanlarınızla sohbet edin! OpenAI ve ChromaDB kullanan modern bir RAG (Retrieval Augmented Generation) uygulaması.

## 🎯 Özellikler

- 📄 **PDF ve CSV Desteği** - Birden fazla dosya yükleyebilirsiniz
- 🔍 **Akıllı Arama** - ChromaDB ile vektör tabanlı arama
- 🤖 **OpenAI Entegrasyonu** - GPT-4o-mini ile doğal cevaplar
- ✂️ **Otomatik Chunking** - Metinleri optimal parçalara böler
- 📊 **Kaynak Gösterme** - Cevapların hangi bölümlerden geldiğini gösterir
- 🎨 **Kullanıcı Dostu Arayüz** - Streamlit ile modern web arayüzü
- 🚫 **Akıllı Reddetme** - Dokümanla alakasız soruları reddeder

## 🛠️ Teknolojiler

- **LangChain** - RAG pipeline yönetimi
- **OpenAI** - Embeddings ve GPT-4o-mini
- **ChromaDB** - Vektör veritabanı
- **Streamlit** - Web arayüzü
- **PyPDF2** - PDF okuma
- **Pandas** - CSV işleme

## 📦 Kurulum

### 1. Projeyi Klonlayın

```bash
git clone https://github.com/KULLANICI_ADINIZ/rag-chat-app.git
cd rag-chat-app
```

### 2. Virtual Environment Oluşturun

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### 4. .env Dosyasını Ayarlayın

`.env` dosyası oluşturun ve OpenAI API key'inizi ekleyin:

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxx
CHROMA_DB_PATH=./data/chroma_db
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

OpenAI API key almak için: https://platform.openai.com/api-keys

## 🚀 Kullanım

### Uygulamayı Başlatın

```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak `http://localhost:8501` açılacak.

### Adımlar

1. **Sol sidebar'dan dosya yükleyin** (PDF veya CSV)
2. **"🚀 Dokümanları İşle"** butonuna tıklayın
3. **Bekleyin** (Embeddings oluşturulacak)
4. **Soru sorun!**

### Örnek Sorular

✅ **Dokümanla ilgili:**
- "Bu dokümanda ne anlatılıyor?"
- "Ana konular neler?"
- "X hakkında ne bilgi var?"

❌ **Alakasız sorular reddedilir:**
- "Hava durumu nasıl?"
- "Yemek tarifi ver"

## 📁 Proje Yapısı

```
rag-chat-app/
├── app.py                    # Ana Streamlit uygulaması
├── requirements.txt          # Python bağımlılıkları
├── .env                      # API anahtarları (gitignore'da)
├── .gitignore               
├── README.md
├── utils/
│   ├── __init__.py
│   ├── document_loader.py   # PDF/CSV okuma
│   ├── chunking.py          # Metin bölme
│   ├── embeddings.py        # OpenAI embeddings
│   └── vector_store.py      # ChromaDB işlemleri
└── data/
    ├── uploads/             # Yüklenen dosyalar
    └── chroma_db/           # ChromaDB veritabanı
```

## 🔧 Chunking Stratejisi

- **Chunk Size:** 500 karakter
- **Overlap:** 50 karakter
- **Metod:** Recursive Character Text Splitter
- **Ayraçlar:** Paragraf → Cümle → Kelime → Karakter

## 🤖 RAG Pipeline

```
1. Dosya Yükleme (PDF/CSV)
2. Metin Çıkarma (DocumentLoader)
3. Chunking (TextChunker)
4. Embedding Oluşturma (OpenAI)
5. ChromaDB'ye Kaydetme (VectorStore)
6. Kullanıcı Sorusu
7. Query Embedding
8. Benzer Chunk'ları Bulma (Similarity Search)
9. Context Oluşturma
10. GPT-4o-mini ile Cevap Üretme
```

## ⚙️ Yapılandırma

### Chunk Ayarları

`.env` dosyasında ayarlayabilirsiniz:

```env
CHUNK_SIZE=500        # Chunk boyutu (karakter)
CHUNK_OVERLAP=50      # Örtüşme miktarı
```

### Embedding Modeli

`utils/embeddings.py` içinde değiştirilebilir:

```python
model_name = "text-embedding-3-small"  # veya "text-embedding-3-large"
```

### Similarity Search

`app.py` içinde `top_k` değiştirebilirsiniz:

```python
results = vector_store.search(query_embedding, top_k=5)
```

## 📊 Performans

- **Embedding Boyutu:** 1536 boyutlu vektör
- **Arama Hızı:** ~100ms (ChromaDB)
- **Cevap Süresi:** 2-5 saniye (OpenAI API)

## 🐛 Sorun Giderme

### "OPENAI_API_KEY bulunamadı" hatası
- `.env` dosyasının proje kök dizininde olduğundan emin olun
- API key'in doğru kopyalandığını kontrol edin

### "ChromaDB oluşturulamadı" hatası
- `data/chroma_db` klasörünün yazma izni olduğunu kontrol edin
- Klasörü silip yeniden oluşturun

### Cevap gelmiyor
- Chunk sayısını kontrol edin (İstatistikler bölümü)
- Daha spesifik sorular sorun
- `top_k` değerini artırın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 👤 Yazar

**[Seyit Ali ÇEVİK]**
- GitHub: [@seyitcvk](https://github.com/seyitcvk)
- Email: seyitcevik00@gmail.com

## 🙏 Teşekkürler

- [LangChain](https://langchain.com/)
- [OpenAI](https://openai.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)


---

⭐ Beğendiyseniz yıldız vermeyi unutmayın!
"""
Document Loader Module
PDF ve CSV dosyalarını okuyup text'e çevirir
"""

import PyPDF2
import pandas as pd
from typing import List, Dict
import os


class DocumentLoader:
    """PDF ve CSV dosyalarını yükler"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.csv']
    
    def load_document(self, file_path: str) -> Dict[str, any]:
        """
        Dosyayı yükler ve içeriğini döndürür
        
        Args:
            file_path: Dosya yolu
            
        Returns:
            Dict: {'content': str, 'metadata': dict}
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self._load_pdf(file_path)
        elif file_extension == '.csv':
            return self._load_csv(file_path)
        else:
            raise ValueError(f"Desteklenmeyen format: {file_extension}")
    
    def _load_pdf(self, file_path: str) -> Dict[str, any]:
        """PDF dosyasını okur"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Tüm sayfaları oku
                text_content = []
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():  # Boş sayfa değilse
                        text_content.append(page_text)
                
                full_text = "\n\n".join(text_content)
                
                metadata = {
                    'source': file_path,
                    'file_type': 'pdf',
                    'num_pages': len(pdf_reader.pages),
                    'total_chars': len(full_text)
                }
                
                return {
                    'content': full_text,
                    'metadata': metadata
                }
                
        except Exception as e:
            raise Exception(f"PDF okuma hatası: {str(e)}")
    
    def _load_csv(self, file_path: str) -> Dict[str, any]:
        """CSV dosyasını okur"""
        try:
            # CSV'yi oku
            df = pd.read_csv(file_path)
            
            # Her satırı text'e çevir
            text_rows = []
            for idx, row in df.iterrows():
                # Satırı string'e çevir
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
                text_rows.append(row_text)
            
            full_text = "\n".join(text_rows)
            
            metadata = {
                'source': file_path,
                'file_type': 'csv',
                'num_rows': len(df),
                'num_columns': len(df.columns),
                'columns': list(df.columns),
                'total_chars': len(full_text)
            }
            
            return {
                'content': full_text,
                'metadata': metadata
            }
            
        except Exception as e:
            raise Exception(f"CSV okuma hatası: {str(e)}")
    
    def load_multiple_documents(self, file_paths: List[str]) -> List[Dict[str, any]]:
        """
        Birden fazla dosyayı yükler
        
        Args:
            file_paths: Dosya yolları listesi
            
        Returns:
            List[Dict]: Her dosya için {'content': str, 'metadata': dict}
        """
        documents = []
        for file_path in file_paths:
            try:
                doc = self.load_document(file_path)
                documents.append(doc)
            except Exception as e:
                print(f"Hata - {file_path}: {str(e)}")
        
        return documents


# Test fonksiyonu
if __name__ == "__main__":
    loader = DocumentLoader()
    
    # Test için örnek kullanım
    print("Document Loader test modunda çalışıyor...")
    print("Kullanım örneği:")
    print("loader = DocumentLoader()")
    print("doc = loader.load_document('dosya.pdf')")
    print("print(doc['content'])")
    
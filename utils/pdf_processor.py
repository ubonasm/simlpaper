import io
from typing import BinaryIO
import PyPDF2

def extract_text_from_pdf(file: BinaryIO) -> str:
    """
    PDFファイルからテキストを抽出
    
    Args:
        file: アップロードされたPDFファイル
        
    Returns:
        抽出されたテキスト
    """
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        text = ""
        
        # 全ページからテキストを抽出
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        # 空白の正規化
        text = " ".join(text.split())
        
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

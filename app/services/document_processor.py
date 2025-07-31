import base64
import requests
import re
from io import BytesIO
from pypdf import PdfReader
import docx

class DocumentProcessor:
    CHUNK_SIZE = 500  # tokens
    CHUNK_OVERLAP = 50 # tokens

    def process_document(self, document: dict) -> list[str]:
        """Orchestrates the document processing pipeline for a single document."""
        content = document.get('content')
        # The new schema passes metadata with filename info
        metadata = document.get('metadata', {})
        filename = metadata.get('filename', '') # Assuming filename is passed in metadata

        with self._get_content_stream(content) as stream:
            text = self._extract_text(stream, filename)
            cleaned_text = self._clean_text(text)
            chunks = self._chunk_text(cleaned_text)
            return chunks

    def _get_content_stream(self, content: str) -> BytesIO:
        """Retrieves content as a stream from a URL or a Base64 string."""
        if content.startswith(('http://', 'https://')):
            # Stream content from URL to avoid loading large files into memory
            response = requests.get(content, timeout=15, stream=True)
            response.raise_for_status()
            return BytesIO(response.content) # Still reads into memory, but can be adapted
        else:
            # Assume Base64 content
            decoded_content = base64.b64decode(content)
            return BytesIO(decoded_content)

    def _extract_text(self, content_stream: BytesIO, filename: str) -> str:
        """Extracts text from a document stream based on its filename extension."""
        file_ext = filename.split('.')[-1].lower() if '.' in filename else ''

        text = ""
        if file_ext == 'pdf':
            reader = PdfReader(content_stream)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif file_ext == 'docx':
            doc = docx.Document(content_stream)
            text = "\n".join(para.text for para in doc.paragraphs)
        else: # Default to plain text for .txt and other unknown types
            text = content_stream.read().decode('utf-8', errors='ignore')
        
        return text

    def _clean_text(self, text: str) -> str:
        """Cleans and normalizes the extracted text."""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace chars with a single space
        text = re.sub(r'(\n)+', '\n', text) # Replace multiple newlines with a single one
        return text

    def _chunk_text(self, text: str) -> list[str]:
        """Splits text into semantic chunks based on word count."""
        words = text.split()
        if not words:
            return []

        chunks = []
        for i in range(0, len(words), self.CHUNK_SIZE - self.CHUNK_OVERLAP):
            chunk_words = words[i:i + self.CHUNK_SIZE]
            chunks.append(" ".join(chunk_words))
        
        return chunks

import asyncio
from app.services.document_processor import DocumentProcessor
from app.services.gemini_service import GeminiPolicyProcessor
from app.services.vector_store_service import VectorStoreService
from app.api.schemas.evaluation import HackRxRequest

class QAService:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.gemini_service = GeminiPolicyProcessor()

    async def answer_questions(self, request: HackRxRequest) -> list[str]:
        """Orchestrates the Q&A process for a document and a list of questions."""
        # 1. Process the document: download, extract text, and chunk it.
        # The document processor expects a dict format similar to our old schema.
        document_dict = {
            'content': request.documents, 
            'metadata': {'filename': request.documents.split('/')[-1].split('?')[0]}
        }
        text_chunks = self.document_processor.process_document(document_dict)

        if not text_chunks:
            raise ValueError("Could not extract any text from the document.")

        # 2. Create a vector store for the document chunks.
        vector_store = VectorStoreService(dimension=768) # Gemini embedding dimension
        embeddings = await self.gemini_service.generate_embeddings_batch(text_chunks)
        vector_store.add_documents([{'text': chunk, 'embedding': emb} for chunk, emb in zip(text_chunks, embeddings)])

        # 3. For each question, find relevant chunks and generate an answer.
        answer_coroutines = []
        for question in request.questions:
            answer_coroutines.append(self._answer_single_question(question, vector_store))
        
        answers = await asyncio.gather(*answer_coroutines)
        return answers

    async def _answer_single_question(self, question: str, vector_store: VectorStoreService) -> str:
        """Generates an answer for a single question using semantic search and an LLM."""
        # a. Find relevant chunks from the document.
        question_embedding = await self.gemini_service.generate_embeddings(question, task_type="retrieval_query")
        search_results = vector_store.search(question_embedding, top_k=5)
        relevant_chunks = [result['text'] for result in search_results]

        # b. Ask the LLM to generate a concise answer based on the chunks.
        answer = await self.gemini_service.generate_answer_from_context(question, relevant_chunks)
        return answer

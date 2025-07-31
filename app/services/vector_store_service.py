import faiss
import numpy as np

class VectorStoreService:
    def __init__(self, dimension: int):
        """Initializes an in-memory FAISS index."""
        self.dimension = dimension
        # Using a flat L2 index for exact search. It's simple and effective for this use case.
        self.index = faiss.IndexFlatL2(dimension)
        self.document_chunks = []

    def add_documents(self, documents: list[dict]):
        """
        Adds document embeddings to the FAISS index.
        Each document should be a dictionary with 'text' and 'embedding' keys.
        """
        if not documents:
            return

        # Store the original text chunks in the same order
        self.document_chunks.extend([doc['text'] for doc in documents])

        # Extract embeddings and convert them to a NumPy array of type float32
        embeddings = np.array([doc['embedding'] for doc in documents]).astype('float32')
        
        # Add the embeddings to the FAISS index
        self.index.add(embeddings)

    def search(self, query_embedding: list, top_k: int) -> list[dict]:
        """
        Searches the index for the most similar document chunks.
        Returns a list of dictionaries, each containing the text and similarity score.
        """
        if self.index.ntotal == 0:
            return []

        # Convert the query embedding to a NumPy array
        query_vector = np.array([query_embedding]).astype('float32')

        # Perform the search
        distances, indices = self.index.search(query_vector, top_k)

        # Process and return the results
        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i != -1:  # FAISS returns -1 if no neighbors are found
                results.append({
                    'text': self.document_chunks[i],
                    'score': float(dist) # Lower distance means more similar
                })
        return results

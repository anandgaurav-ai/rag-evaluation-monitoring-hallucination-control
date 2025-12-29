from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, data_dir = "data/raw_docs"):
        self.data_dir = Path(data_dir)
        self.embed_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        embedding_dim = self.embed_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.texts = []
        self.sources = []

        self._build_index()

    def _build_index(self):
        all_embeddings = []

        for file_path in self.data_dir.glob("*"):
            if file_path.suffix not in {".txt", ".pdf"}:
                continue

            if file_path.suffix == ".txt":
                text = file_path.read_text(encoding = "utf-8", errors = "ignore")

            elif file_path.suffix == ".pdf":
                from pypdf import PdfReader

                reader = PdfReader(file_path)
                text = " ".join(
                    page.extract_text() or "" for page in reader.pages
                )

            chunks = self._chunk_text(text)
            if not chunks:
                continue

            embeddings = self.embed_model.encode(
                chunks,
                show_progress_bar = False
            )

            all_embeddings.extend(embeddings)
            self.texts.extend(chunks)
            self.sources.extend([file_path.name] * len(chunks))

        if not all_embeddings:
            raise ValueError("No documents found to build the index.")

        self.index.add(np.array(all_embeddings).astype("float32"))


    def _chunk_text(self, text, size = 300, overlap = 50):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + size])
            start += size - overlap
        return chunks


    def retrieve(self, query, top_k = 3):
        """
        Returns:
            context (str): concatenated retrieved chunks
            retrieval_score (float): normalized relevance score (0-1)
            sources (list[str]): source document names
        """

        # Embed query
        query_embedding = self.embed_model.encode(
            [query], show_progress_bar = False
        ).astype("float32")

        # Search FAISS
        distances, indices = self.index.search(query_embedding, top_k)

        indices = indices[0]
        distances = distances [0]

        # Gather retrieved chunks
        retrieved_chunks = []
        retrieved_sources = []

        for idx in indices:
            if idx == -1:
                continue
            retrieved_chunks.append(self.texts[idx])
            retrieved_sources.append(self.texts[idx])

        # Build context
        context = "\n\n".join(retrieved_chunks)


        # Convert distance to similarity score
        # L2 distance -> similarity (simple normalization)
        avg_distance = float(np.mean(distances))
        retrieval_score = 1/(1+avg_distance)

        return context, retrieval_score, retrieved_sources
    


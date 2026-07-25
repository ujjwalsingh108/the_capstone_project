# RAG Workflow

Recommended retrieval flow:

1. Ingest documents from market reports, internal knowledge bases, and operational data.
2. Chunk the documents and build embeddings.
3. Store embeddings in a vector database.
4. Retrieve top-k context for each prediction request.
5. Pass the retrieved evidence into the LLM prompt alongside structured features.

The scaffold currently uses an in-memory retriever so the application can run before the production retrieval layer is implemented.

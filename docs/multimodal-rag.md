# Multimodal RAG

This project is a **multimodal Retrieval-Augmented Generation** system:

- **Multimodal** — it uses images (fashion photos) and text (names, prices, links).
- **Retrieval-augmented** — ChromaDB returns the nearest catalog outfit before the model writes.
- **Generation** — Llama 4 Maverick Vision Instruct produces the style report.

Foundation models do not contain this catalog by default. RAG merges retrieved
SKU data with the model's visual understanding so answers stay specific: real
item names, prices, and purchase links instead of a generic caption.

## Pipeline

1. **Multimodal input processing** — the upload is encoded to a visual vector
   and a Base64 JPEG for the LLM.
2. **Vector-based retrieval** — ChromaDB cosine search finds the nearest catalog
   embedding; related SKUs are loaded with a metadata filter on `image_url`.
3. **Context-enhanced generation** — retrieved facts are injected into the
   prompt so the writeup is catalog-grounded, not image-only.

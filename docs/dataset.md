# Dataset notes

The demo catalog is curated from [taylorswiftstyle.com](https://taylorswiftstyle.com).
Each row is one fashion item with name, price, purchase link, and the full-outfit
image it appears in.

## How the catalog was built

1. **Web scraping** — the first 10 pages were collected with BeautifulSoup:
   outfit photos, item names, and prices.
2. **Storage** — records were stored in a pandas DataFrame.
3. **Image encoding** — each photo was stored as JPEG Base64 (for the vision LLM)
   and as a ResNet50 vector (for nearest-neighbor retrieval).
4. **Vector index** — `scripts/ingest_chromadb.py` upserts those vectors into a
   persistent **ChromaDB** collection (`fashion_outfits`) with cosine space.

## Download and ingest

```bash
bash scripts/download_dataset.sh
python scripts/ingest_chromadb.py
```

This saves `data/swift-style-embeddings.pkl` (gitignored) and writes the Chroma
index under `data/chroma/` (also gitignored).

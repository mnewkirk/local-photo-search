# local-photo-search

A fully local, offline photo search engine. Search your photo library by person, place, description, or color — without sending anything to the cloud.

## Features (planned)

- **Semantic search** — describe what you're looking for in natural language ("kids playing in the park") powered by CLIP embeddings
- **Face recognition** — find photos of specific people, with auto-clustering and named references
- **Place search** — search by location using GPS/EXIF data with offline reverse geocoding
- **Color search** — find photos by dominant color palette
- **Scene descriptions** — auto-generated natural language descriptions via LLaVA (Ollama)
- **Non-destructive** — photos are never modified; all metadata lives in a separate SQLite database

## Stack

- **Python 3.10+**
- **SQLite + sqlite-vec** — fast, portable database with vector similarity search
- **CLIP** (via open-clip) — image embeddings for semantic search
- **LLaVA** (via Ollama) — local vision model for generating scene descriptions
- **face_recognition** (dlib) — face detection, encoding, and matching
- **ARW support** — extracts metadata from Sony RAW files alongside JPEGs

## Target deployment

Proof of concept on macOS, production target is a **UGREEN NAS** (Intel N100, 8GB RAM) running in Docker.

## Project structure

```
local-photo-search/
├── photosearch/          # Main Python package
│   ├── __init__.py
│   ├── index.py          # Indexing pipeline (EXIF, CLIP, faces, LLaVA, color)
│   ├── db.py             # Database schema and queries
│   ├── search.py         # Search logic (semantic, face, place, color)
│   ├── faces.py          # Face detection, encoding, clustering
│   ├── clip_embed.py     # CLIP embedding generation
│   ├── describe.py       # LLaVA scene description
│   ├── colors.py         # Dominant color extraction
│   └── exif.py           # EXIF/GPS extraction
├── cli.py                # CLI entry point
├── requirements.txt
├── Dockerfile            # (future)
├── docker-compose.yml    # (future)
└── README.md
```

## Quick start

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Make sure Ollama is running with LLaVA
ollama pull llava

# Index a folder of photos
python cli.py index /path/to/photos

# Search
python cli.py search --query "sunset at the beach"
python cli.py search --person "Matt"
python cli.py search --color "blue"
```

## Development milestones

1. **M1** — EXIF + CLIP indexing + color extraction → semantic search works
2. **M2** — Face detection + clustering
3. **M3** — LLaVA descriptions via Ollama
4. **M4** — Full CLI with all search modes
5. **M5** — Scale test (196 photos)
6. **M6** — Web UI
7. **M7** — Docker packaging for NAS

## License

Private project.

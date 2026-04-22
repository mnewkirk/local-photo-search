"""local-photo-search: fully local photo search by person, place, description, and color."""

__version__ = "0.1.0"

# Register HEIF/HEIC opener with PIL at import time so every code path
# (indexer, CLIP embed, thumbnailing, face detection) can open
# iPhone/iPad .heic files without special-casing them. pillow-heif is a
# hard requirement in requirements.txt; the try/except only guards
# against broken installs so imports don't fail catastrophically.
try:
    from pillow_heif import register_heif_opener as _register_heif_opener
    _register_heif_opener()
except ImportError:
    pass

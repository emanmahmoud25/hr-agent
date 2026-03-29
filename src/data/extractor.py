"""PDF / TXT → clean text extraction."""
from pathlib import Path
import fitz   # pymupdf


def extract_text(filename: str, data: bytes) -> str:
    """Extract text from raw bytes (used by FastAPI upload)."""
    if filename.lower().endswith(".pdf"):
        doc  = fitz.open(stream=data, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    elif filename.lower().endswith(".txt"):
        return data.decode("utf-8", errors="ignore").strip()
    raise ValueError(f"Unsupported file type: {filename}")


def load_cv(path: str | Path) -> str:
    """Load CV from file path (used by scripts / widget)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() == ".pdf":
        doc  = fitz.open(str(p))
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    elif p.suffix.lower() == ".txt":
        return p.read_text(encoding="utf-8", errors="ignore").strip()
    raise ValueError(f"Unsupported file type: {p.suffix}")


def load_all_cvs(extracted_dir: Path) -> list[dict]:
    """Load all cleaned CVs from extracted_cvs/ folder."""
    cvs = []
    for txt_file in sorted(extracted_dir.glob("*.txt")):
        parts    = txt_file.name.split("_", 1)
        position = parts[0] if len(parts) > 1 else "Unknown"
        cvs.append({
            "filename" : txt_file.name,
            "position" : position,
            "file_path": txt_file,
            "text"     : txt_file.read_text(encoding="utf-8", errors="ignore"),
        })
    return cvs

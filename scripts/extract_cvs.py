"""
Script: Extract raw PDFs → clean text files.
Usage: python scripts/extract_cvs.py
"""
from pathlib import Path
from src.config import cfg
from src.data.extractor import load_cv


def main():
    raw_dir  = cfg.RAW_CV_DIR
    out_dir  = cfg.EXTRACTED_DIR
    pdf_files= list(raw_dir.glob("*.pdf")) + list(raw_dir.glob("*.txt"))

    if not pdf_files:
        print(f"No files found in {raw_dir}")
        return

    print(f"Found {len(pdf_files)} files in {raw_dir}")
    ok, fail = 0, 0

    for pdf in pdf_files:
        try:
            text     = load_cv(pdf)
            out_file = out_dir / (pdf.stem + ".txt")
            out_file.write_text(text, encoding="utf-8")
            print(f"  ✅ {pdf.name} → {len(text)} chars")
            ok += 1
        except Exception as e:
            print(f"  ❌ {pdf.name}: {e}")
            fail += 1

    print(f"\nDone: {ok} ok | {fail} failed → {out_dir}")


if __name__ == "__main__":
    main()

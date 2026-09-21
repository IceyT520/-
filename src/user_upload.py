"""用户上传文献处理: 保存PDF -> 文本抽取/清洗/分块 -> 嵌入入库。

用户上传的文献与核心库共用同一个 Chroma 集合, 通过元数据 lib="user"
区分(核心库记录无此字段或 lib="core"), 检索时统一参与混合排序。
上传记录登记在 data/upload_registry.json。
"""

import json
import time
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import clean_pages_paged, extract_pages, normalize_formulas, _ODD_SPACES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
REGISTRY_PATH = PROJECT_ROOT / "data" / "upload_registry.json"


def _load_registry() -> list[dict]:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return []


def _save_registry(records: list[dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def list_uploads() -> list[dict]:
    """已上传文献列表(新的在前)。"""
    return sorted(_load_registry(), key=lambda r: r["uploaded_at"], reverse=True)


def _guess_title(text: str, fallback: str) -> str:
    """取首页首个足够长的行作为标题, 失败则用文件名。"""
    for line in text.splitlines():
        s = line.strip()
        if 15 < len(s) < 200 and not s.lower().startswith(
            ("downloaded", "http", "www.", "received")
        ):
            return s
    return fallback


def process_pdf(pdf_path: Path, splitter: RecursiveCharacterTextSplitter) -> dict:
    """处理单个PDF, 返回 {title, records(含页码元数据), n_pages}。"""
    pages = extract_pages(pdf_path)
    page_texts = [normalize_formulas(t) for t in clean_pages_paged(pages)]
    full_text = "\n\n".join(page_texts)
    title = _guess_title(full_text, pdf_path.stem)
    doc_id = f"user-{int(time.time())}-{pdf_path.stem[:30]}"
    records = []
    for pno, ptext in enumerate(page_texts, 1):
        if len(ptext.strip()) < 30:
            continue
        for i, chunk in enumerate(splitter.split_text(ptext)):
            records.append(
                {
                    "id": f"{doc_id}#p{pno}-{i}",
                    "text": chunk,
                    "source": f"user/{pdf_path.name}",
                    "title": title,
                    "lib": "user",
                    "page": pno,
                }
            )
    return {"title": title, "records": records, "n_pages": len(pages)}


def save_and_register(pdf_path: Path, n_chunks: int, title: str) -> None:
    registry = _load_registry()
    registry.append(
        {
            "filename": pdf_path.name,
            "title": title,
            "n_chunks": n_chunks,
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    _save_registry(registry)

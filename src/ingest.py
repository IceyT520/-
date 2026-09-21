"""PDF 文本抽取、清洗、化学式归一化与语义分块。

用法:
    python src/ingest.py [--pdf-dir pdfs] [--out data/chunks.jsonl]

流程:
    1. 扫描 pdf 目录, 用 PyMuPDF 按阅读顺序抽取文本
    2. 清洗: 去除下载水印、页眉页脚(跨页重复行)、孤立页码
    3. 化学式归一化: Unicode 上下标 -> ASCII (Li₃YCl₆ -> Li3YCl6)
    4. RecursiveCharacterTextSplitter 分块 (600字符/重叠120, 段落优先)
    5. 输出 JSONL, 每块带 source(文件名) 与 title(论文标题) 元数据
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Unicode 上下标 -> ASCII, 统一化学式与单位写法, 保证检索时格式一致
_SUB_SUPER = str.maketrans(
    "₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺₋₊",
    "01234567890123456789-+-+",
)

_WATERMARK_PATTERNS = [
    re.compile(r"downloaded from", re.IGNORECASE),
    re.compile(r"^(at\s+)?beijing university of chemical technology", re.IGNORECASE),
    re.compile(r"^by beijing univ", re.IGNORECASE),
    re.compile(r"https?://doi\.org/", re.IGNORECASE),
]
_PAGE_NUMBER = re.compile(r"^\d{1,4}$")
# 特殊空格(全角/em/窄不换行等)统一为普通空格, 否则水印与化学式匹配会漏
_ODD_SPACES = str.maketrans("     　", "      ")


def normalize_formulas(text: str) -> str:
    return text.translate(_SUB_SUPER)


def extract_pages(pdf_path: Path, engine: str = "text") -> list[str]:
    if engine == "md":
        import pymupdf4llm

        doc = fitz.open(pdf_path)
        n = len(doc)
        doc.close()
        return [
            pymupdf4llm.to_markdown(str(pdf_path), pages=[i]) for i in range(n)
        ]
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text", sort=True))
    doc.close()
    return pages


def clean_pages(pages: list[str]) -> str:
    """去水印/页眉页脚/页码, 并把同段落的断行合并。"""
    pages = [p.translate(_ODD_SPACES) for p in pages]
    # 统计跨页重复行(页眉页脚特征): 出现在超过 30% 页面且较短的行
    line_page_count: dict[str, int] = {}
    for page in pages:
        for line in set(l.strip() for l in page.splitlines()):
            if 3 < len(line) < 80:
                line_page_count[line] = line_page_count.get(line, 0) + 1
    n_pages = max(len(pages), 1)
    repeated = {
        line for line, cnt in line_page_count.items() if cnt / n_pages > 0.3
    }

    cleaned_pages = []
    for page in pages:
        kept = []
        for line in page.splitlines():
            s = line.strip()
            if not s:
                kept.append("")
                continue
            if _PAGE_NUMBER.match(s):
                continue
            if s in repeated:
                continue
            if any(p.search(s) for p in _WATERMARK_PATTERNS):
                continue
            kept.append(s)
        cleaned_pages.append("\n".join(kept))

    full = "\n\n".join(cleaned_pages)
    # 段内断行合并: 单行换行(前后非空行)合并为空格; 保留空行作为段落边界
    full = re.sub(r"(?<=\S)\n(?=\S)", " ", full)
    # 连字符断词: "conduct-\nivity" 已在上面合并为 "conduct- ivity" 的情况较少见, 不强行处理
    full = re.sub(r"[ \t]+", " ", full)
    full = re.sub(r"\n{3,}", "\n\n", full)
    return _strip_watermark_fragments(full)


# 出版商在 PDF 中加入的下载水印常被双栏排版打碎、甚至嵌入零宽空格,
# 行级过滤无法清除, 合并后统一按模式剔除
_POST_MERGE_WATERMARKS = [
    # ACS: 含零宽空格的 article-pdf 链接及其后的机构署名
    re.compile(r"pubs\.\u200b?acs\.\u200b?org/\S+?\.pdf\s*(by\s+BEIJING\s+UNIV[^\n]{0,60})?"),
    # Science: Downloaded from https://... at Beijing University... on ...
    re.compile(r"Downloaded from\s*https?://\S+\s+at(\s+Beijing\s+University[^\n]{0,60}?\d{4}|\s*)"),
    # RSC 被打碎的残片: "Downloaded from by of on"
    re.compile(r"Downloaded from by of on\s*"),
    # Wiley: onlinelibrary 链接 + "by Beijing University of Chemistry Technology..."
    re.compile(r"\S*wiley\.com/\S+\s+by\s+Beijing\s+University\s+of\s+Chemistry\s+Technology[^\n]{0,120}"),
    re.compile(r"by\s+Beijing\s+University\s+of\s+Chemistry\s+Technology[^\n]{0,120}"),
    # Science 残余碎片: "Beijing University of on 18,"
    re.compile(r"Beijing\s+University\s+of\s+on\s+\d{1,2},?\s*"),
    # Wiley 残片: "Chemistry (Wiley Online Library) on [date]. See the Terms and Conditions (...)"
    re.compile(r"(of\s+)?Chemistry\s+(Wiley\s+Online\s+Library\s+)?on\s+\[\d{2}/\d{2}/\d{4}\]\.?\s*See\s+the\s+Terms\s+and\s+Conditions\s*\(\S*\)\s*(on\s*)?"),
    # 残余的机构署名碎片(含与正文粘连的情况, 如 "BEIJING UNIVamorphous")
    re.compile(r"(by\s+)?BEIJING\s+UNIV(\s+OF\s+CHEMICAL\s+TECHNOLOGY)?(\s+user)?(\s+on\s+September(\s+\d{1,2},?\s*\d{4})?)?\s*"),
    # "Downloaded fromACCESS" 这类与后文粘连的残片
    re.compile(r"Downloaded from(?=\w)"),
]


def _strip_watermark_fragments(text: str) -> str:
    text = text.replace("​", "")
    for pat in _POST_MERGE_WATERMARKS:
        text = pat.sub(" ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def load_titles(pdf_dir: Path) -> dict[str, str]:
    """从 manifest.csv 读取 文件名->标题 映射。"""
    titles = {}
    manifest = pdf_dir / "manifest.csv"
    if manifest.exists():
        with open(manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("filename") and row.get("title"):
                    titles[row["filename"]] = row["title"]
    return titles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default=str(PROJECT_ROOT / "pdfs"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "chunks.jsonl"))
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument(
        "--engine", choices=["text", "md"], default="text",
        help="text=PyMuPDF纯文本(默认, 快); md=pymupdf4llm转Markdown(慢, 表格/版面更好)",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"未在 {pdf_dir} 找到 PDF 文件")

    titles = load_titles(pdf_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for pdf in pdfs:
            pages = extract_pages(pdf, engine=args.engine)
            text = normalize_formulas(clean_pages(pages))
            chunks = splitter.split_text(text)
            title = titles.get(pdf.name, pdf.stem)
            for i, chunk in enumerate(chunks):
                record = {
                    "id": f"{pdf.stem}#{i}",
                    "text": chunk,
                    "source": pdf.name,
                    "title": title,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_chunks += len(chunks)
            print(f"{pdf.name}: {len(pages)} 页 -> {len(chunks)} 块")

        # 人工整理的结构化文档(如综述表格), 弥补双栏PDF表格抽取错乱的问题。
        # 按行切分(一行一材料一块)并前缀表头: 保证"某材料的某指标"类精确查询
        # 能命中独立行, 而不是被同块内其他材料稀释向量信号
        curated_dir = PROJECT_ROOT / "data" / "curated"
        for txt in sorted(curated_dir.glob("*.txt")):
            text = normalize_formulas(
                txt.read_text(encoding="utf-8").translate(_ODD_SPACES)
            )
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            title = lines[0][:80]
            header = "\n".join(lines[:2])
            n = 0
            for i, line in enumerate(lines[2:], 0):
                if line.startswith("注:") or len(line) < 10:
                    continue
                record = {
                    "id": f"curated-{txt.stem}#{i}",
                    "text": f"{header}\n{line}",
                    "source": f"curated/{txt.name}",
                    "title": title,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
            total_chunks += n
            print(f"curated/{txt.name}: -> {n} 块(按行)")

    print(f"\n共处理 {len(pdfs)} 篇文献, 生成 {total_chunks} 个文本块 -> {out_path}")


if __name__ == "__main__":
    main()

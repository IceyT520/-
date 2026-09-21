"""参考文献格式化导出: GB/T 7714 与 BibTeX。

数据来源: pdfs/manifest.csv 的 DOI; 两篇核心文献与人工整理表手动补充。
格式化通过 doi.org 内容协商完成(CSL: china-national-standard-gb-t-7714-2015),
结果缓存在 data/citation_cache.json, 避免重复请求。
"""

import csv
import html
import json
import re
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "data" / "citation_cache.json"

# manifest 之外的核心文献 DOI 补充
EXTRA_DOIS = {
    "Li_Du_2025_ACSNano_HalideSSE_Review.pdf": "10.1021/acsnano.4c15005",
    "Zhao_Sun_2025_Science_Oxyhalide_13.7mS.pdf": "10.1126/science.adt9678",
}

# 人工整理数据表的引用(指向原综述)
CURATED_CITATION = (
    "卤化物固态电解质性能数据汇总表（项目组人工整理自: Li C, Du Y. Building a Better "
    "All-Solid-State Lithium-Ion Battery with Halide Solid-State Electrolyte[J]. "
    "ACS Nano, 2025, 19(4): 4121-4155, Table 1. DOI: 10.1021/acsnano.4c15005）"
)


def _load_source_dois() -> dict[str, str]:
    dois = dict(EXTRA_DOIS)
    manifest = PROJECT_ROOT / "pdfs" / "manifest.csv"
    if manifest.exists():
        with open(manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("filename") and row.get("doi"):
                    dois[row["filename"]] = row["doi"]
    return dois


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _fetch_citation(doi: str, fmt: str) -> str:
    """获取格式化引用。fmt: 'gbt7714' | 'bibtex'
    BibTeX 走 doi.org 内容协商; GB/T 7714 由 Crossref 元数据自行组装
    (doi.org 的 CSL 服务不支持国标样式, 返回406)。"""
    if fmt == "bibtex":
        req = urllib.request.Request(
            f"https://doi.org/{doi}", headers={"Accept": "application/x-bibtex"}
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read().decode("utf-8").strip()
    return _format_gbt7714(_fetch_crossref(doi))


def _fetch_crossref(doi: str) -> dict:
    req = urllib.request.Request(
        f"https://api.crossref.org/works/{doi}",
        headers={"User-Agent": "hsse-rag-qa/1.0 (mailto:2119007310@qq.com)"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))["message"]


def _clean_meta_text(s: str) -> str:
    """清理 Crossref 元数据文本: 去除 <sub>/<sup>/<i>/<inf> 等排版标签并反转义实体。
    开始标签连同其前导空白删除(使 'Li <sub>3</sub>' 合为 'Li3'),
    闭合标签只删标签本身; 再合并标签在化学式中留下的断口('Li3 YCl6' -> 'Li3YCl6')。"""
    s = re.sub(r"\s*<[^>]+>", "", s)
    s = re.sub(r"</[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    # 数字与后续元素符号之间的空格是标签断口, 合并('Li3 YCl6' -> 'Li3YCl6');
    # 后接小写开头单词(如 and)的不受影响
    s = re.sub(r"(?<=\d)\s+(?=[A-Z][a-z]?)", "", s)
    return s


def _format_gbt7714(meta: dict) -> str:
    """GB/T 7714-2015 期刊论文: 主要责任者. 题名[J]. 刊名, 年, 卷(期): 页码."""
    authors = []
    for a in meta.get("author", [])[:3]:
        family = a.get("family", "")
        given = "".join(p[0] for p in a.get("given", "").replace("-", " ").split() if p)
        authors.append(f"{family} {given}".strip())
    if len(meta.get("author", [])) > 3:
        authors.append("等")
    author_str = ", ".join(authors) or "佚名"

    title = _clean_meta_text((meta.get("title") or [""])[0])
    journal = _clean_meta_text((meta.get("container-title") or [""])[0])
    year = (meta.get("issued", {}).get("date-parts") or [[None]])[0][0] or ""
    volume = meta.get("volume", "")
    issue = meta.get("issue", "")
    page = meta.get("page", "")
    vol_issue = volume + (f"({issue})" if issue else "")
    tail = ", ".join(str(x) for x in [year, vol_issue] if x)
    if page:
        tail += f": {page}"
    return f"{author_str}. {title}[J]. {journal}, {tail}."


def export_citations(sources: list[str], fmt: str = "gbt7714") -> str:
    """把一组来源文件名导出为指定格式的参考文献列表(按来源去重)。"""
    dois = _load_source_dois()
    cache = _load_cache()
    seen: set[str] = set()
    lines = []

    for src in sources:
        if src in seen:
            continue
        seen.add(src)

        if src.startswith("curated/"):
            lines.append(CURATED_CITATION if fmt != "bibtex"
                         else "@misc{hsse_curated_table1, title={卤化物固态电解质性能数据汇总表}, "
                              "note={人工整理自 Li & Du, ACS Nano 2025, Table 1}}")
            continue
        if src.startswith("user/"):
            lines.append(f"[用户上传文献] {src.removeprefix('user/')}")
            continue

        doi = dois.get(src)
        if not doi:
            lines.append(f"[未找到DOI, 请手动核对] {src}")
            continue

        key = f"{doi}|{fmt}"
        if key not in cache:
            try:
                cache[key] = _fetch_citation(doi, fmt)
                time.sleep(0.5)
            except Exception as e:
                lines.append(f"[获取失败:{e}, 请手动核对] DOI: {doi}")
                continue
        lines.append(cache[key])

    _save_cache(cache)
    return "\n\n".join(lines) if lines else "(无引用来源)"

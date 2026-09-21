"""化学感知高级查询: 按元素/材料类别/性能阈值筛选材料。

数据源: data/curated/ 下人工整理的性能数据表(逐行解析为结构化记录)。
覆盖问题类型:
- "列出库里所有氧卤化物"          -> 按类别筛选
- "含In的氯化物有哪些"            -> 按元素+类别筛选
- "离子电导率超过5 mS/cm的材料"   -> 按数值阈值筛选
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURATED_DIR = PROJECT_ROOT / "data" / "curated"

_ROW_RE = re.compile(
    r"^(?P<formula>\S+?)(?:\s*[(（](?P<note>[^)）]*)[)）])?\s*—\s*"
    r"离子电导率 (?P<ionic>\S+)(?: mS cm-1)? \| "
    r"活化能 (?P<ea>\S+)(?: eV)? \| "
    r"电子电导率 (?P<electronic>\S+)(?: S cm-1)? \| "
    r"ESW (?P<esw>.*)$"
)

_HALOGENS = {"Cl", "Br", "I", "F"}
# 常见元素中文名 -> 元素符号
_ELEMENT_ZH = {
    "锂": "Li", "钠": "Na", "钾": "K", "镁": "Mg", "钙": "Ca",
    "钇": "Y", "钪": "Sc", "铟": "In", "镧": "La", "钽": "Ta",
    "铌": "Nb", "锆": "Zr", "铪": "Hf", "钬": "Ho", "镱": "Yb",
    "铒": "Er", "铽": "Tb", "镥": "Lu", "镓": "Ga", "铝": "Al",
    "铁": "Fe", "锌": "Zn", "锡": "Sn", "钛": "Ti",
}
_CATEGORY_KEYWORDS = {
    "氧卤化物": "oxyhalide", "氧氯化物": "oxyhalide", "oxyhalide": "oxyhalide",
    "氯化物": "chloride", "溴化物": "bromide", "氟化物": "fluoride",
    "碘化物": "iodide", "混合卤素": "mixed",
}


def _to_float(s: str) -> float | None:
    """解析 '5.5×10-4' / '1.171' / '~10-10' / '-' 等数值写法。"""
    s = s.strip().lstrip("~")
    if s in {"-", ""}:
        return None
    m = re.match(r"^([\d.]+)(?:×10(-?\d+))?$", s)
    if not m:
        return None
    base = float(m.group(1))
    if m.group(2):
        base *= 10 ** int(m.group(2))
    return base


def _elements_of(formula: str) -> set[str]:
    return set(re.findall(r"[A-Z][a-z]?", formula))


def category_of(formula: str) -> str:
    els = _elements_of(formula)
    halogens = els & _HALOGENS
    if "O" in els and halogens:
        return "oxyhalide"
    if len(halogens) > 1:
        return "mixed"
    if halogens == {"Cl"}:
        return "chloride"
    if halogens == {"Br"}:
        return "bromide"
    if halogens == {"F"}:
        return "fluoride"
    if halogens == {"I"}:
        return "iodide"
    return "other"


def load_materials() -> list[dict]:
    materials = []
    if not CURATED_DIR.exists():
        return materials
    for txt in sorted(CURATED_DIR.glob("*.txt")):
        for line in txt.read_text(encoding="utf-8").splitlines():
            m = _ROW_RE.match(line.strip())
            if not m:
                continue
            d = m.groupdict()
            materials.append(
                {
                    "formula": d["formula"],
                    "note": d.get("note") or "",
                    "ionic": d["ionic"],
                    "ionic_val": _to_float(d["ionic"]),
                    "ea": d["ea"],
                    "electronic": d["electronic"],
                    "esw": d["esw"].strip(),
                    "category": category_of(d["formula"]),
                    "elements": _elements_of(d["formula"]),
                    "source": f"curated/{txt.name}",
                }
            )
    return materials


def try_compare_query(query: str) -> str | None:
    """识别并回答材料对比查询(如"对比Li3YCl6和Li3InCl6的电导率和活化能");
    数据全部来自人工校准性能表, 不匹配或材料不足则返回 None(走常规RAG)。"""
    if not re.search(r"对比|比较|相比|哪个|vs|VS", query):
        return None
    from qa import extract_formulas  # 延迟导入避免循环依赖

    formulas = extract_formulas(query, max_n=5)
    if len(formulas) < 2:
        return None

    materials = load_materials()
    rows = [x for x in materials if x["formula"] in formulas]
    found = {x["formula"] for x in rows}
    missing = [f for f in formulas if f not in found]
    if len(found) < 2:
        return None

    table_rows = [
        "| 材料 | 离子电导率 (mS cm-1) | 活化能 (eV) | 电子电导率 (S cm-1) | ESW |",
        "|---|---|---|---|---|",
    ]
    for x in rows:
        note = f"（{x['note']}）" if x["note"] else ""
        table_rows.append(
            f"| {x['formula']}{note} | {x['ionic']} | {x['ea']} | {x['electronic']} | {x['esw']} |"
        )

    # 离子电导率排序结论(按各材料最高报道值)
    best: dict[str, float] = {}
    for x in rows:
        if x["ionic_val"] is not None:
            best[x["formula"]] = max(best.get(x["formula"], 0), x["ionic_val"])
    conclusion = ""
    if len(best) >= 2:
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        conclusion = (
            "\n\n**离子电导率排序(取各材料最高报道值)**: "
            + " > ".join(f"{f}({v:g})" for f, v in ranked)
            + " mS cm-1"
        )

    missing_str = (
        f"\n\n注: 性能表暂未收录 {', '.join(missing)}, 可尝试常规提问获取文献信息。"
        if missing else ""
    )
    return (
        f"以下对比数据全部来自人工整理的性能数据表 [1]:\n\n"
        + "\n".join(table_rows)
        + conclusion
        + missing_str
        + "\n\n参考文献:\n[1] 卤化物固态电解质性能数据汇总表 (人工整理自 Li & Du, ACS Nano 2025, Table 1)"
    )


def try_chem_query(query: str) -> str | None:
    """识别并回答化学感知类列举查询; 不匹配则返回 None(走常规RAG)。"""
    if not re.search(r"列出|哪些|所有|都有", query):
        return None

    materials = load_materials()
    if not materials:
        return None

    conditions: list[str] = []

    # 类别条件
    category = None
    for kw, cat in _CATEGORY_KEYWORDS.items():
        if kw in query:
            category = cat
            conditions.append(kw)
            break

    # 元素条件: "含In" / "含铟" / "有Ta"
    elements = []
    zh_query = query
    for zh, sym in _ELEMENT_ZH.items():
        if f"含{zh}" in zh_query or f"有{zh}" in zh_query:
            elements.append(sym)
            zh_query = zh_query.replace(zh, "")
    for m in re.finditer(r"[含有]([A-Z][a-z]?)", zh_query):
        elements.append(m.group(1))
    for e in elements:
        conditions.append(f"含{e}")

    # 数值条件: "电导率超过/大于/高于 X"
    threshold = None
    m = re.search(r"电导率(?:超过|大于|高于|>|>=)\s*([\d.]+)", query)
    if m:
        threshold = float(m.group(1))
        conditions.append(f"离子电导率 > {threshold} mS cm-1")

    if not (category or elements or threshold is not None):
        return None

    result = materials
    if category:
        result = [x for x in result if x["category"] == category]
    for e in elements:
        result = [x for x in result if e in x["elements"]]
    if threshold is not None:
        result = [
            x for x in result
            if x["ionic_val"] is not None and x["ionic_val"] > threshold
        ]

    cond_str = "、".join(conditions)
    if not result:
        return f"在人工整理的性能数据表中未找到符合「{cond_str}」的材料。"
    rows = [
        "| 材料 | 离子电导率 (mS cm-1) | 活化能 (eV) | 电子电导率 (S cm-1) | ESW |",
        "|---|---|---|---|---|",
    ]
    for x in result:
        note = f"（{x['note']}）" if x["note"] else ""
        rows.append(
            f"| {x['formula']}{note} | {x['ionic']} | {x['ea']} | {x['electronic']} | {x['esw']} |"
        )
    return (
        f"人工整理的性能数据表中共有 **{len(result)} 种**符合「{cond_str}」的材料 [1]:\n\n"
        + "\n".join(rows)
        + "\n\n参考文献:\n[1] 卤化物固态电解质性能数据汇总表 (人工整理自 Li & Du, ACS Nano 2025, Table 1)"
    )

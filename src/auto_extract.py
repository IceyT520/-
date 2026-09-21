"""知识库自生长: 用 LLM 从新上传论文中抽取材料性能数据。

流程: 上传PDF -> 全文(截断)送 DeepSeek -> 按人工表行格式抽取 -> 校验 ->
存入待确认队列(data/pending_materials.json) -> 管理员确认后并入
data/curated/user_contributed.txt 并向量化入库。
"""

import json
import time
from pathlib import Path

from materials_db import _ROW_RE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PENDING_PATH = PROJECT_ROOT / "data" / "pending_materials.json"
USER_CURATED = PROJECT_ROOT / "data" / "curated" / "user_contributed.txt"
USER_CURATED_HEADER = (
    "用户上传文献性能数据表 (LLM抽取+人工确认, 格式同人工校准表)\n"
    "每行一种材料: 材料式 — 室温离子电导率(RT ionic conductivity) | "
    "活化能(activation energy) | 电子电导率(electronic conductivity) | 电化学稳定窗口(ESW)"
)

_EXTRACT_PROMPT = """你是材料数据抽取助手。从以下卤化物固态电解质论文文本中, 抽取材料的电化学性能数据。
对每种材料输出一行, 严格使用此格式:
材料式 — 离子电导率 <数值> mS cm-1 | 活化能 <数值> eV | 电子电导率 <数值> S cm-1 | ESW <数值>

规则:
- 只抽取文本中明确给出的数值, 不得推测或编造;
- 某项指标文中没有就写 -;
- 化学式一律用 ASCII 形式 (如 Li3YCl6, LiTaOCl4);
- 离子电导率统一换算为 mS cm-1 的数值(若原文为 S cm-1 请换算);
- 只输出数据行, 不要任何解释、表头或序号;
- 若文本中没有任何可抽取的数据, 输出: 无"""


def extract_from_text(text: str, llm_client, max_chars: int = 12000) -> list[str]:
    """从论文文本中抽取性能数据行(已按人工表格式校验)。"""
    resp = llm_client.chat.completions.create(
        model="deepseek-chat",
        temperature=0,
        messages=[
            {"role": "system", "content": _EXTRACT_PROMPT},
            {"role": "user", "content": text[:max_chars]},
        ],
    )
    content = resp.choices[0].message.content.strip()
    if content == "无":
        return []
    rows = []
    for line in content.splitlines():
        line = line.strip().strip("`").lstrip("-•*0123456789.、) ")
        if _ROW_RE.match(line):
            rows.append(line)
    return rows


def load_pending() -> list[dict]:
    if PENDING_PATH.exists():
        return json.loads(PENDING_PATH.read_text(encoding="utf-8"))
    return []


def save_pending(rows: list[dict]) -> None:
    PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    PENDING_PATH.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def add_pending(new_rows: list[str], source_file: str) -> int:
    pending = load_pending()
    existing = {p["row"] for p in pending}
    added = 0
    for row in new_rows:
        if row not in existing:
            pending.append(
                {
                    "row": row,
                    "source_file": source_file,
                    "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            added += 1
    save_pending(pending)
    return added


def confirm_pending() -> list[str]:
    """把全部待确认行追加到 user_contributed.txt, 返回确认的行列表。"""
    pending = load_pending()
    if not pending:
        return []
    if not USER_CURATED.exists():
        USER_CURATED.parent.mkdir(parents=True, exist_ok=True)
        USER_CURATED.write_text(USER_CURATED_HEADER + "\n", encoding="utf-8")
    with open(USER_CURATED, "a", encoding="utf-8") as f:
        for p in pending:
            f.write(p["row"] + "\n")
    rows = [p["row"] for p in pending]
    save_pending([])
    return rows

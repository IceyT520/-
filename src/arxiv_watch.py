"""每周新文献追踪: 从 arXiv 检索卤化物固态电解质最新论文。

用法:
    python src/arxiv_watch.py --fetch          # 立即拉取(服务器由 systemd timer 每周执行)
    python src/arxiv_watch.py --fetch --days 7
结果写入 data/arxiv_new.json, 网页端读取展示并可一键下载入库。
"""

import argparse
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "data" / "arxiv_new.json"

_NS = {"atom": "http://www.w3.org/2005/Atom"}

# 单一查询(左结合): (卤素 AND 固态电解质) OR 卤化物超离子导体 OR 卤化物固态电解质
_SEARCH_QUERY = (
    "all:halide+AND+all:%22solid+electrolyte%22"
    "+OR+all:%22halide+superionic+conductor%22"
    "+OR+all:%22halide+solid-state+electrolyte%22"
)


def _query_api() -> list:
    url = (
        "https://export.arxiv.org/api/query?search_query="
        + _SEARCH_QUERY
        + "&sortBy=submittedDate&sortOrder=descending&max_results=40"
    )
    # arXiv 对频繁请求会返回 406 软封禁, 指数退避重试
    for attempt, wait in enumerate([0, 8, 20]):
        if wait:
            time.sleep(wait)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                root = ET.fromstring(resp.read().decode("utf-8"))
            return root.findall("atom:entry", _NS)
        except urllib.error.HTTPError as e:
            if e.code != 406 or attempt == 2:
                raise
    return []


def fetch(days: int = 14) -> list[dict]:
    seen: set[str] = set()
    entries = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    for e in _query_api():
        arxiv_id = e.find("atom:id", _NS).text.rsplit("/", 1)[-1]
        if arxiv_id in seen:
            continue
        published = e.find("atom:published", _NS).text
        dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        if dt < cutoff:
            continue
        seen.add(arxiv_id)
        title = re.sub(r"\s+", " ", e.find("atom:title", _NS).text).strip()
        summary = re.sub(r"\s+", " ", e.find("atom:summary", _NS).text).strip()
        entries.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "published": published[:10],
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
                "summary": summary[:300] + ("..." if len(summary) > 300 else ""),
            }
        )

    entries.sort(key=lambda x: x["published"], reverse=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "days": days,
                "entries": entries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return entries


def load_entries() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text(encoding="utf-8"))
    return {"fetched_at": None, "days": 14, "entries": []}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--days", type=int, default=14)
    args = parser.parse_args()
    if args.fetch:
        entries = fetch(args.days)
        print(f"最近 {args.days} 天新文献: {len(entries)} 篇 -> {OUT_PATH}")
        for e in entries[:5]:
            print(f"  [{e['published']}] {e['title'][:70]}")


if __name__ == "__main__":
    main()

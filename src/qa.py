"""问答主模块: 语义检索 + DeepSeek 生成带引用的答案。

用法:
    python src/qa.py "Li3YCl6的室温离子电导率是多少?"   # 单次提问
    python src/qa.py                                     # 交互模式
    python src/qa.py --no-llm "问题"                     # 只检索不生成(调试用)

DeepSeek API Key 放在项目根目录 .env 文件中: DEEPSEEK_API_KEY=sk-...
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 嵌入/精排模型均已缓存时离线加载, 避免启动时向 huggingface.co 发送检查请求的长重试
_HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
if (_HF_HUB / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2").exists() \
        and (_HF_HUB / "models--BAAI--bge-reranker-base").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

from dotenv import load_dotenv  # noqa: E402

from build_db import COLLECTION_NAME, EMBEDDING_MODEL  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path("data/chroma_db")  # 相对路径, 见 RAGEngine 中的说明


def curated_sources() -> list[str]:
    """data/curated/ 下所有人工整理文档的 source 标识(与 ingest.py 一致)。"""
    d = PROJECT_ROOT / "data" / "curated"
    return [f"curated/{p.name}" for p in sorted(d.glob("*.txt"))] if d.exists() else []

SYSTEM_PROMPT = """你是卤化物固态电解质领域的科研问答助手。请严格遵守以下规则:
1. 仅依据提供的文献片段回答, 不得使用片段之外的知识;
2. 若片段中没有回答问题所需的信息, 请明确说明"未在文献中找到该数据", 不得编造数值或结论;
3. 答案中每个论断后用 [编号] 标注其来源片段, 如 [1]、[2];
4. 答案末尾列出参考文献列表, 格式: [编号] 论文标题;
5. 化学式统一写作 ASCII 数字形式 (如 Li3YCl6), 数值保留单位;
6. 用中文回答。"""

REWRITE_PROMPT = """你是查询改写助手。根据对话历史, 把用户的最新问题改写为一个独立问题。
规则: 只补全省略的主语/指代; 不要添加原问题中没有的限定条件(合成方法、时间等); 保留化学式。

示例1:
历史: [用户] Li3YCl6的室温离子电导率是多少? [助手] 球磨法合成的Li3YCl6为0.51×10^-3 S cm^-1。
最新问题: 那它的活化能呢?
输出: Li3YCl6的活化能是多少?

示例2:
历史: [用户] 氯化物和溴化物的电化学窗口分别是多少? [助手] 氯化物约0.6-4.3V, 溴化物约1.5-3.4V。
最新问题: 那碘化物呢?
输出: 碘化物固态电解质的电化学窗口是多少?

只输出改写后的问题本身, 不要任何解释。"""

# 与 ingest.py 一致的化学式归一化, 保证查询与语料格式匹配
_SUB_SUPER = str.maketrans(
    "₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺₋₊",
    "01234567890123456789-+-+",
)


def normalize_query(query: str) -> str:
    return query.translate(_SUB_SUPER)


# 化学式识别: 元素符号+数字的组合且含卤素, 如 Li3YCl6 / LiTaOCl4 / Li3Ta3O4Cl10
# (?<![A-Za-z0-9]) 而非 \b: Python 的 \b 把中文字符也算作单词字符, 会漏匹配
_FORMULA_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Z][a-z]?\d*\.?\d*){2,8}(?![A-Za-z0-9])")
_HALOGENS = ("Cl", "Br", "I", "F")


def extract_formulas(query: str, max_n: int = 3) -> list[str]:
    """从问题中提取化学式(用于关键词精确匹配, 弥补向量检索对化学式不敏感的问题)。"""
    found = []
    for m in _FORMULA_RE.finditer(normalize_query(query)):
        token = m.group(0)
        if any(c.isdigit() for c in token) and any(h in token for h in _HALOGENS):
            if token not in found:
                found.append(token)
    return found[:max_n]


class RAGEngine:
    def __init__(self, db_path: Path | str = DB_PATH, top_k: int = 4):
        import chromadb
        from sentence_transformers import SentenceTransformer

        # chromadb 1.x Rust 后端对非ASCII绝对路径有bug, 统一切换到项目根目录用相对路径
        os.chdir(PROJECT_ROOT)
        db_path = Path(db_path)
        if db_path.is_absolute():
            try:
                db_path = db_path.relative_to(PROJECT_ROOT)
            except ValueError:
                pass
        if not db_path.exists():
            raise SystemExit("向量库不存在, 请先运行 src/ingest.py 和 src/build_db.py")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(db_path))
        self.collection = client.get_collection(COLLECTION_NAME)
        self.top_k = top_k

    def add_documents(self, records: list[dict], batch_size: int = 64) -> None:
        """把新文本块嵌入并写入向量库(与 build_db 一致的标题拼接策略)。"""
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            embeddings = self.model.encode(
                [f"{r['title']}\n{r['text']}" for r in batch],
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            self.collection.add(
                ids=[r["id"] for r in batch],
                embeddings=embeddings.tolist(),
                documents=[r["text"] for r in batch],
                metadatas=[
                    {"source": r["source"], "title": r["title"],
                     "lib": r.get("lib", "core")}
                    for r in batch
                ],
            )

    def count_chunks(self, lib: str | None = None) -> int:
        """文本块总数; lib="user" 时只统计用户上传部分。"""
        if lib is None:
            return self.collection.count()
        return len(self.collection.get(where={"lib": lib}, include=[])["ids"])

    def _vector_rank(self, q_emb: list, n: int, where_document=None, where=None) -> list[dict]:
        kwargs = dict(
            query_embeddings=q_emb,
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        if where_document:
            kwargs["where_document"] = where_document
        if where:
            kwargs["where"] = where
        result = self.collection.query(**kwargs)
        hits = []
        for doc_id, doc, meta, dist in zip(
            result["ids"][0], result["documents"][0],
            result["metadatas"][0], result["distances"][0],
        ):
            hits.append(
                {"id": doc_id, "text": doc, "source": meta["source"],
                 "title": meta["title"], "similarity": 1 - dist}
            )
        return hits

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """混合检索: 向量语义排序 + 化学式精确匹配, RRF 融合。

        向量模型对化学式不敏感(Li3YCl6/Li3ScCl6 语义几乎相同),
        因此对问题中识别出的每个化学式, 在包含该式子的文本块内做向量排序,
        与全局向量排序做倒数排名融合(RRF)。
        """
        k = top_k or self.top_k
        q_emb = self.model.encode(
            [normalize_query(query)], normalize_embeddings=True
        ).tolist()

        rank_lists = [(1.0, self._vector_rank(q_emb, max(k * 5, 20)))]
        for formula in extract_formulas(query):
            rank_lists.append(
                (2.0, self._vector_rank(q_emb, 10, where_document={"$contains": formula}))
            )
            # 结构化数据分支: 人工校准数据表是人工核验过的事实源,
            # 对"某材料某指标"类查询给最高投票权重, 确保其进入精排候选池
            rank_lists.append(
                (
                    2.5,
                    self._vector_rank(
                        q_emb, 3,
                        where_document={"$contains": formula},
                        where={"source": {"$in": curated_sources()}},
                    ),
                )
            )

        rrf_score: dict[str, float] = {}
        best: dict[str, dict] = {}
        for weight, hits in rank_lists:
            for rank, h in enumerate(hits):
                rrf_score[h["id"]] = rrf_score.get(h["id"], 0.0) + weight / (60 + rank + 1)
                if h["id"] not in best or h["similarity"] > best[h["id"]]["similarity"]:
                    best[h["id"]] = h
        ordered = sorted(rrf_score, key=lambda i: rrf_score[i], reverse=True)
        candidates = [best[i] for i in ordered[: max(k * 3, 12)]]

        # 精排结果作为"第三位投票者"再次RRF融合, 而非直接覆盖:
        # 通用精排模型偏爱关键词密集的文本(如数据表), 直接覆盖会压制
        # 化学式精确匹配与向量语义两个领域信号的排序结果
        reranked = self._rerank(query, candidates)
        final_score: dict[str, float] = {}
        for rank, h in enumerate(reranked):
            final_score[h["id"]] = 1.5 / (60 + rank + 1)
        for weight, hits in rank_lists:
            for rank, h in enumerate(hits):
                if h["id"] in final_score:
                    final_score[h["id"]] += weight / (60 + rank + 1)
        ordered = sorted(final_score, key=lambda i: final_score[i], reverse=True)
        # 同源多样性约束: 每个来源最多占2席, 防止关键词密集的单一来源
        # (如人工数据表)挤占全部席位, 保证答案引用的来源多样性
        picked: list[dict] = []
        per_source: dict[str, int] = {}
        for i in ordered:
            src = best[i]["source"]
            if per_source.get(src, 0) >= 2:
                continue
            per_source[src] = per_source.get(src, 0) + 1
            picked.append(best[i])
            if len(picked) >= k:
                break
        return picked

    @property
    def reranker(self):
        """交叉编码精排模型(懒加载)。对粗排结果按 (问题, 文本块) 对重新打分,
        显著提升复杂问法的排序质量, 是主流RAG产品的标配环节。"""
        if not hasattr(self, "_reranker"):
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder("BAAI/bge-reranker-base")
        return self._reranker

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return candidates
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

    def build_prompt(self, query: str, hits: list[dict]) -> str:
        parts = ["以下是与问题相关的文献片段:\n"]
        for i, h in enumerate(hits, 1):
            parts.append(f"[{i}] (来源: {h['title']})\n{h['text']}\n")
        parts.append(f"\n问题: {query}")
        return "\n".join(parts)

    def _llm_client(self):
        if not hasattr(self, "_client"):
            load_dotenv(PROJECT_ROOT / ".env")
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise SystemExit("未找到 DEEPSEEK_API_KEY, 请在 .env 中配置")
            from openai import OpenAI

            self._client = OpenAI(
                api_key=api_key, base_url="https://api.deepseek.com"
            )
        return self._client

    def _messages(self, query: str, hits: list[dict]) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.build_prompt(query, hits)},
        ]

    def rewrite_query(self, query: str, history: list | None) -> str:
        """多轮对话: 把指代性追问改写为独立问题(无历史则原样返回)。"""
        if not history:
            return query

        def _text(content) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # Gradio 消息块格式
                return " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            return str(content)

        recent = [
            {"role": h["role"], "content": _text(h.get("content"))[:300]}
            for h in history[-4:]
            if h.get("role") in ("user", "assistant")
        ]
        if not recent or not any(r["content"].strip() for r in recent):
            return query
        resp = self._llm_client().chat.completions.create(
            model="deepseek-chat",
            temperature=0,
            max_tokens=200,
            messages=[{"role": "system", "content": REWRITE_PROMPT}]
            + recent
            + [{"role": "user", "content": query}],
        )
        rewritten = resp.choices[0].message.content.strip()
        return rewritten or query

    def ask(self, query: str, history: list | None = None) -> dict:
        standalone = self.rewrite_query(query, history)
        from materials_db import try_chem_query, try_compare_query

        chem = try_compare_query(standalone) or try_chem_query(standalone)
        if chem is not None:
            self.last_hits = []
            return {
                "question": query,
                "standalone_query": standalone,
                "answer": chem,
                "references": [],
                "elapsed_s": 0,
            }
        hits = self.retrieve(standalone)
        client = self._llm_client()
        t0 = time.time()
        resp = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.2,
            messages=self._messages(standalone, hits),
        )
        elapsed = time.time() - t0
        answer = resp.choices[0].message.content
        return {
            "question": query,
            "standalone_query": standalone,
            "answer": answer,
            "references": [
                {"n": i, "title": h["title"], "source": h["source"]}
                for i, h in enumerate(hits, 1)
            ],
            "elapsed_s": round(elapsed, 1),
        }

    def ask_stream(self, query: str, history: list | None = None):
        """流式生成, 供网页界面使用。逐段产出答案文本, 最后产出耗时标记。
        检索结果存入 self.last_hits, 供界面展示引用片段。"""
        standalone = self.rewrite_query(query, history)
        from materials_db import try_chem_query, try_compare_query

        chem = try_compare_query(standalone) or try_chem_query(standalone)
        if chem is not None:
            self.last_hits = []
            yield chem + "\n\n---\n(本回答由化学感知查询直接生成, 数据来自人工校准性能表)"
            return
        hits = self.retrieve(standalone)
        self.last_hits = hits
        client = self._llm_client()
        t0 = time.time()
        stream = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.2,
            messages=self._messages(standalone, hits),
            stream=True,
        )
        partial = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                partial += delta
                yield partial
        elapsed = time.time() - t0
        tail = f"\n\n---\n生成耗时 {elapsed:.1f}s | 引用来源见上方参考文献列表"
        if standalone != query:
            tail += f"\n(追问已改写为: {standalone})"
        yield partial + tail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="提问内容, 省略则进入交互模式")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--no-llm", action="store_true", help="只检索, 不调用大模型")
    parser.add_argument("--json", action="store_true", help="以JSON输出完整结果")
    args = parser.parse_args()

    engine = RAGEngine(top_k=args.top_k)

    def run(query: str) -> None:
        if args.no_llm:
            hits = engine.retrieve(query)
            for i, h in enumerate(hits, 1):
                print(f"\n[{i}] {h['title']} ({h['source']}) 相似度={h['similarity']:.3f}")
                print(h["text"][:300] + ("..." if len(h["text"]) > 300 else ""))
            return
        result = engine.ask(query)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(result["answer"])
            print(f"\n(生成耗时 {result['elapsed_s']}s)")

    if args.query:
        run(args.query)
    else:
        print("卤化物固态电解质问答系统 (输入 exit 退出)")
        while True:
            try:
                query = input("\n问题> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if query.lower() in {"exit", "quit", ""}:
                break
            run(query)


if __name__ == "__main__":
    main()

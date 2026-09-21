"""向量知识库构建: 文本块 -> 多语言嵌入 -> Chroma 持久化库。

用法:
    python src/build_db.py [--chunks data/chunks.jsonl] [--db data/chroma_db]

嵌入模型: paraphrase-multilingual-MiniLM-L12-v2 (跨语言检索, CPU 可运行)
支持中文提问检索英文文献语料。模型从 hf-mirror.com 下载(国内镜像)。
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from sentence_transformers import SentenceTransformer  # noqa: E402

import chromadb  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "hsse"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--db", default="data/chroma_db")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    # chromadb 1.x Rust 后端对非ASCII绝对路径有bug(索引文件写不出), 一律用相对路径
    os.chdir(PROJECT_ROOT)

    records = []
    with open(args.chunks, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    if not records:
        raise SystemExit(f"{args.chunks} 为空, 请先运行 src/ingest.py")
    print(f"载入 {len(records)} 个文本块")

    print(f"加载嵌入模型 {EMBEDDING_MODEL} (首次运行需下载约470MB)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=args.db)
    # 重建集合, 保证与 chunks.jsonl 一致
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        # 嵌入时拼接论文标题: 标题含关键化学式与主题词, 显著提升小模型对
        # "某化合物/某方法"类问题的区分度; 库中存储的仍是原文
        embeddings = model.encode(
            [f"{r['title']}\n{r['text']}" for r in batch],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        collection.add(
            ids=[r["id"] for r in batch],
            embeddings=embeddings.tolist(),
            documents=[r["text"] for r in batch],
            metadatas=[
                {k: v for k, v in
                 {"source": r["source"], "title": r["title"], "page": r.get("page")}.items()
                 if v is not None}
                for r in batch
            ],
        )
        print(f"已写入 {min(start + args.batch_size, len(records))}/{len(records)}")

    print(f"\n向量库构建完成: {collection.count()} 条记录 -> {args.db}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
课设 RAG - 知识库构建脚本（断点续传 + 自动重试）
读取 kb_source/ 下所有 md -> 切块 -> bge-m3 分批向量化 -> FAISS 本地索引
特性:
  - 每批向量化后增量落盘，中断后重跑自动从上次进度继续
  - embedding 请求失败自动重试（指数退避）
用法: python build_index.py [--sample N] [--reset]
  --sample N  只取前 N 个文件（调试用）
  --reset     清空缓存重新全量构建
"""
import os
import sys
import glob
import time
import argparse
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import faiss
import numpy as np

BASE = Path(__file__).resolve().parent
KB_SOURCE = BASE / "kb_source"
INDEX_DIR = BASE / "index"

EMBED_MODEL = "bge-m3"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
BATCH = 256
MAX_RETRY = 6
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

CACHE_VEC = INDEX_DIR / "kb_vecs_cache.npy"    # 已算出的向量
CACHE_CNT = INDEX_DIR / "kb_vecs_done.txt"     # 已算出的向量数


def embed_with_retry(embeddings, batch, attempt=1):
    """带指数退避重试的 embedding 调用"""
    try:
        return embeddings.embed_documents(batch)
    except Exception as e:
        if attempt > MAX_RETRY:
            raise
        wait = min(2 ** attempt, 60)
        print(f"\n   [重试] 第 {attempt} 次失败: {type(e).__name__} {str(e)[:120]}  "
              f"{wait}s 后重试...", flush=True)
        time.sleep(wait)
        return embed_with_retry(embeddings, batch, attempt + 1)


def load_documents(sample=None):
    files = sorted(glob.glob(str(KB_SOURCE / "**" / "*.md"), recursive=True))
    if sample:
        files = files[:sample]
    docs = []
    for fp in files:
        try:
            text = Path(fp).read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not text:
            continue
        rel = os.path.relpath(fp, KB_SOURCE)
        docs.append(Document(page_content=text, metadata={"source": rel}))
    return docs


def save_cache(vecs, done):
    with open(CACHE_VEC, "wb") as f:
        np.save(f, np.array(vecs, dtype="float32"), allow_pickle=False)
    CACHE_CNT.write_text(str(done), encoding="utf-8")


def load_cache():
    if CACHE_VEC.exists() and CACHE_CNT.exists():
        with open(CACHE_VEC, "rb") as f:
            vecs = np.load(f)
        done = int(CACHE_CNT.read_text(encoding="utf-8").strip())
        if len(vecs) == done:
            return vecs.tolist(), done
    return [], 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    # 首次运行时 index/ 可能不存在；缓存写入发生在向量化阶段，须提前建目录
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset:
        for p in (CACHE_VEC, CACHE_CNT):
            p.unlink(missing_ok=True)
        print("[reset] 已清空缓存")

    if not KB_SOURCE.exists():
        print(f"[错误] 知识库源目录不存在: {KB_SOURCE}")
        print("       请先准备 kb_source/（放入 .md 文档）后再运行。")
        sys.exit(1)

    print("1) 加载文档...")
    docs = load_documents(args.sample)
    if not docs:
        print(f"[错误] 未在 {KB_SOURCE} 下找到任何非空 .md 文档，无法构建索引。")
        sys.exit(1)
    print(f"   共加载 {len(docs)} 个 md 文档")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"2) 切块完成，共 {len(chunks)} 个文本块")

    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]
    n = len(texts)

    print(f"3) 初始化 embedding: {EMBED_MODEL} (Ollama)...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE)

    # 断点续传
    all_vecs, done0 = load_cache()
    print(f"4) 分批向量化（batch={BATCH}），共 {n} 块，已有缓存 {done0} 块")
    t0 = time.time()
    i = done0
    try:
        while i < n:
            batch = texts[i:i + BATCH]
            vecs = embed_with_retry(embeddings, batch)
            all_vecs.extend(vecs)
            i = min(i + BATCH, n)
            # 每批落盘（增量检查点）
            save_cache(all_vecs, i)
            el = time.time() - t0
            rate = (i - done0) / el if el > 0 else 0
            eta = (n - i) / rate / 60 if rate > 0 else 0
            print(f"   进度 {i}/{n}  ({i/n*100:.1f}%)  "
                  f"{rate:.0f}块/s  剩余约 {eta:.1f} 分钟", end="\r", flush=True)
    except KeyboardInterrupt:
        print(f"\n[中断] 已保存 {i} 块缓存，重跑可从断点继续")
        sys.exit(1)
    print(f"\n   向量化完成，共 {len(all_vecs)} 个向量，耗时 {time.time()-t0:.0f}s")

    if not all_vecs:
        print("[错误] 没有生成任何向量，构建中止。")
        sys.exit(1)

    print("5) 构建 FAISS 索引...")
    mat = np.array(all_vecs, dtype="float32")
    index = faiss.IndexFlatIP(mat.shape[1])
    faiss.normalize_L2(mat)
    index.add(mat)
    print(f"   索引向量数: {index.ntotal}, 维度: {mat.shape[1]}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (INDEX_DIR / "kb_index.faiss").write_bytes(faiss.serialize_index(index))
    with open(INDEX_DIR / "kb_metas.npy", "wb") as f:
        np.save(f, np.array(metas, dtype=object), allow_pickle=True)
    with open(INDEX_DIR / "kb_texts.npy", "wb") as f:
        np.save(f, np.array(texts, dtype=object), allow_pickle=True)
    print(f"6) 索引已保存到: {INDEX_DIR}")

    # 清理缓存
    for p in (CACHE_VEC, CACHE_CNT):
        p.unlink(missing_ok=True)
    print("   缓存已清理，构建完成。")


if __name__ == "__main__":
    main()

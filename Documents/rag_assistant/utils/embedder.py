"""
embedder.py
负责：
1. 用 sentence-transformers 将文本块向量化
2. 管理 ChromaDB 本地向量数据库（增删查）
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import hashlib


# ChromaDB 本地持久化目录
CHROMA_DB_PATH = "./chroma_db"


def _get_chroma_client() -> chromadb.PersistentClient:
    """
    获取或创建 ChromaDB 持久化客户端。
    数据保存在本地 ./chroma_db 目录，重启后数据不丢失。
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client


def _get_collection(client: chromadb.PersistentClient, collection_name: str = "rag_docs"):
    """
    获取或创建 ChromaDB 集合（类似数据库中的表）。
    使用余弦相似度作为距离度量，适合语义检索场景。

    Args:
        client:          ChromaDB 客户端
        collection_name: 集合名称

    Returns:
        chromadb.Collection 对象
    """
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # 余弦相似度
    )
    return collection


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    加载 sentence-transformers 多语言嵌入模型。
    paraphrase-multilingual-MiniLM-L12-v2 支持中文、英文等 50+ 种语言。
    首次运行会自动下载模型（约 400MB），之后从本地缓存加载。

    Args:
        model_name: Hugging Face 模型名称

    Returns:
        SentenceTransformer 模型实例
    """
    model = SentenceTransformer(model_name)
    return model


def _make_doc_id(source: str, chunk_id: int) -> str:
    """
    为每个文本块生成唯一 ID（避免重复插入）。
    格式：sha256(source_name)[:8] + "_chunk_{chunk_id}

    Args:
        source:   来源文件名
        chunk_id: 块序号

    Returns:
        唯一字符串 ID
    """
    hash_prefix = hashlib.sha256(source.encode()).hexdigest()[:8]
    return f"{hash_prefix}_chunk_{chunk_id}"


def add_chunks_to_db(
    chunks: List[Dict],
    model: SentenceTransformer,
    collection_name: str = "rag_docs",
) -> int:
    """
    将文本块向量化并批量存入 ChromaDB。

    步骤：
    1. 提取所有 chunk 的文本
    2. 批量调用模型生成 embeddings（向量）
    3. 写入 ChromaDB（自动去重：同 ID 不重复写入）

    Args:
        chunks:          pdf_parser.parse_pdf() 返回的块列表
        model:           已加载的 SentenceTransformer 模型
        collection_name: 存入的 ChromaDB 集合名

    Returns:
        成功写入的块数量
    """
    if not chunks:
        return 0

    client = _get_chroma_client()
    collection = _get_collection(client, collection_name)

    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks]
    ids = [_make_doc_id(c["source"], c["chunk_id"]) for c in chunks]

    # 批量向量化（model 内部自动分批处理）
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    # upsert：存在则更新，不存在则插入
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    return len(chunks)


def query_similar_chunks(
    query: str,
    model: SentenceTransformer,
    top_k: int = 5,
    collection_name: str = "rag_docs",
    source_filter: List[str] = None,
) -> List[Dict]:
    """
    检索与用户问题最相关的文本块。

    步骤：
    1. 将用户问题向量化
    2. 在 ChromaDB 中做余弦相似度搜索
    3. 返回 top_k 个最相关结果

    Args:
        query:           用户问题文本
        model:           已加载的 SentenceTransformer 模型
        top_k:           返回最相关结果数量
        collection_name: 查询的 ChromaDB 集合名
        source_filter:   可选，只在指定文件中搜索（文件名列表）

    Returns:
        List[Dict]，每项包含:
            - text:      匹配到的文本内容
            - source:    来源文件名
            - chunk_id:  块序号
            - distance:  余弦距离（越小越相关）
    """
    client = _get_chroma_client()
    collection = _get_collection(client, collection_name)

    # 集合为空时直接返回
    if collection.count() == 0:
        return []

    query_embedding = model.encode([query], show_progress_bar=False).tolist()

    # 构造过滤条件（按文件名过滤）
    where_clause = None
    if source_filter and len(source_filter) > 0:
        if len(source_filter) == 1:
            where_clause = {"source": {"$eq": source_filter[0]}}
        else:
            where_clause = {"source": {"$in": source_filter}}

    query_params = {
        "query_embeddings": query_embedding,
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where_clause:
        query_params["where"] = where_clause

    results = collection.query(**query_params)

    # 整理成统一格式
    output = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_id": meta.get("chunk_id", -1),
                "distance": round(dist, 4),
            })

    return output


def delete_source_from_db(
    source_name: str,
    collection_name: str = "rag_docs",
) -> int:
    """
    从 ChromaDB 中删除指定 PDF 文件的所有向量数据。

    Args:
        source_name:     要删除的文件名（与存入时的 source 字段一致）
        collection_name: ChromaDB 集合名

    Returns:
        删除操作执行后集合中剩余的文档数量
    """
    client = _get_chroma_client()
    collection = _get_collection(client, collection_name)

    # 按 source metadata 字段过滤删除
    collection.delete(where={"source": {"$eq": source_name}})

    return collection.count()


def get_all_sources(collection_name: str = "rag_docs") -> List[str]:
    """
    获取当前向量库中所有已索引的 PDF 文件名（去重）。

    Args:
        collection_name: ChromaDB 集合名

    Returns:
        文件名列表（已去重）
    """
    client = _get_chroma_client()
    collection = _get_collection(client, collection_name)

    if collection.count() == 0:
        return []

    # 获取所有 metadata
    all_data = collection.get(include=["metadatas"])
    sources = list({m["source"] for m in all_data["metadatas"] if "source" in m})
    return sorted(sources)


def get_chunk_count_per_source(collection_name: str = "rag_docs") -> Dict[str, int]:
    """
    统计每个 PDF 文件在向量库中存储的块数量（用于 UI 展示）。

    Args:
        collection_name: ChromaDB 集合名

    Returns:
        Dict，key 为文件名，value 为块数量
    """
    client = _get_chroma_client()
    collection = _get_collection(client, collection_name)

    if collection.count() == 0:
        return {}

    all_data = collection.get(include=["metadatas"])
    count_map: Dict[str, int] = {}
    for meta in all_data["metadatas"]:
        src = meta.get("source", "unknown")
        count_map[src] = count_map.get(src, 0) + 1

    return count_map

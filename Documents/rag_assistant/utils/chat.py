"""
chat.py
负责：
1. 构建 RAG（检索增强生成）的 prompt
2. 调用 Claude API 生成回答
3. 管理多轮对话历史
"""

import os
import anthropic
from typing import List, Dict, Tuple


def get_anthropic_client() -> anthropic.Anthropic:
    """
    初始化 Anthropic 客户端。
    API Key 从环境变量 ANTHROPIC_API_KEY 读取，不硬编码在代码里。

    Returns:
        anthropic.Anthropic 客户端实例

    Raises:
        ValueError: 若环境变量未设置
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 ANTHROPIC_API_KEY 环境变量，"
            "请在终端执行: export ANTHROPIC_API_KEY='your-key-here'"
        )
    return anthropic.Anthropic(api_key=api_key)


def build_system_prompt() -> str:
    """
    构建系统级 prompt，告诉 Claude 它的角色和行为规范。

    Returns:
        系统 prompt 字符串
    """
    return """你是一位专业的行业研究助手，擅长分析研究报告、白皮书和行业文档。

你的工作方式：
1. 根据用户提供的【参考资料】回答问题，优先使用资料中的信息
2. 回答时引用具体来源，格式为「来源：《文件名》第X段」
3. 如果参考资料不足以完整回答，明确告知用户，并补充你的知识
4. 回答结构清晰，使用条目、小标题等格式提升可读性
5. 使用中文回答（除非用户用其他语言提问）
6. 保持客观、专业，不夸大或歪曲资料内容"""


def build_rag_context(retrieved_chunks: List[Dict]) -> str:
    """
    将检索到的文本块格式化成 Claude 可理解的参考资料块。

    Args:
        retrieved_chunks: embedder.query_similar_chunks() 返回的结果列表

    Returns:
        格式化后的参考资料字符串，直接嵌入 user message
    """
    if not retrieved_chunks:
        return "【当前知识库中未找到相关内容】"

    context_parts = ["【参考资料】（按相关度排序）\n"]
    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk["source"]
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]
        # 相关度：余弦距离越小越好，转换成百分比展示
        relevance = round((1 - chunk["distance"]) * 100, 1)
        context_parts.append(
            f"---\n"
            f"[资料{i}] 来源：《{source}》第{chunk_id + 1}段 "
            f"（相关度：{relevance}%）\n"
            f"{text}\n"
        )

    return "\n".join(context_parts)


def build_user_message(question: str, retrieved_chunks: List[Dict]) -> str:
    """
    将用户问题和检索到的参考资料拼接成完整的 user message。

    Args:
        question:         用户的原始问题
        retrieved_chunks: 检索结果列表

    Returns:
        拼接好的 user message 字符串
    """
    rag_context = build_rag_context(retrieved_chunks)
    return f"{rag_context}\n\n【用户问题】\n{question}"


def chat_with_claude(
    question: str,
    retrieved_chunks: List[Dict],
    conversation_history: List[Dict],
    model_name: str,
    max_tokens: int = 2048,
) -> Tuple[str, List[Dict]]:
    """
    调用 Claude API 进行 RAG 问答，支持多轮对话。

    Args:
        question:              用户当前输入的问题
        retrieved_chunks:      检索到的相关文本块（用于构建上下文）
        conversation_history:  之前的对话历史，格式为 Claude API 的 messages 列表
        model_name:            Claude 模型 ID（如 claude-sonnet-4-20250514）
        max_tokens:            生成回答的最大 token 数

    Returns:
        Tuple:
            - answer (str): Claude 生成的回答文本
            - updated_history (List[Dict]): 加入本轮对话后的完整历史
    """
    client = get_anthropic_client()

    # 构建本轮 user message（包含检索上下文 + 用户问题）
    user_message_content = build_user_message(question, retrieved_chunks)

    # 将本轮消息追加到历史（不修改原始列表）
    messages = list(conversation_history) + [
        {"role": "user", "content": user_message_content}
    ]

    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        system=build_system_prompt(),
        messages=messages,
    )

    answer = response.content[0].text

    # 更新对话历史（保存原始问题，不含检索上下文，保持历史简洁）
    updated_history = list(conversation_history) + [
        {"role": "user", "content": question},          # 历史中只存用户原始问题
        {"role": "assistant", "content": answer},
    ]

    return answer, updated_history


def format_sources_for_display(retrieved_chunks: List[Dict]) -> str:
    """
    将检索到的来源信息格式化成用户友好的 markdown 字符串，
    在界面上展示在 Claude 回答的下方。

    Args:
        retrieved_chunks: 检索结果列表

    Returns:
        markdown 格式的来源摘要字符串
    """
    if not retrieved_chunks:
        return ""

    lines = ["---", "**📚 本次检索参考来源：**"]
    seen = set()
    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk["source"]
        chunk_id = chunk["chunk_id"]
        relevance = round((1 - chunk["distance"]) * 100, 1)
        key = f"{source}_{chunk_id}"
        if key not in seen:
            seen.add(key)
            # 截取预览文本（前80字）
            preview = chunk["text"][:80].replace("\n", " ")
            lines.append(
                f"{i}. **《{source}》** 第{chunk_id + 1}段 "
                f"（相关度 {relevance}%）  \n"
                f"   *预览：{preview}…*"
            )

    return "\n".join(lines)

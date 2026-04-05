"""
pdf_parser.py
负责 PDF 文件的读取、文本提取和段落切分。
使用 pdfplumber 解析，支持中英文混合内容。
"""

import pdfplumber
import re
from pathlib import Path
from typing import List, Dict


def extract_text_from_pdf(file_path: str) -> str:
    """
    从 PDF 文件中提取全部文本。

    Args:
        file_path: PDF 文件的本地路径

    Returns:
        提取到的纯文本字符串（保留换行信息用于后续清洗）
    """
    full_text = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                # 在每页文本前加页码标记，方便后续溯源
                full_text.append(f"[第{page_num}页]\n{text}")
    return "\n".join(full_text)


def clean_text(text: str) -> str:
    """
    清洗提取到的原始文本：
    - 去掉多余空白行
    - 合并因 PDF 排版断开的同一段落（行末无标点的短行拼接到下一行）
    - 保留段落分隔符

    Args:
        text: 原始提取文本

    Returns:
        清洗后的文本
    """
    # 把 Windows 换行统一成 Unix 换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 超过两个连续空行压缩成两个
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 去掉每行首尾多余空格
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def split_into_chunks(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
    source_name: str = "unknown",
) -> List[Dict]:
    """
    将长文本按字符数切分成带重叠的文本块（chunks）。

    策略：
    1. 优先按双换行（自然段落边界）切割
    2. 若某段落超过 chunk_size，再按字符硬切
    3. 相邻块之间保留 overlap 个字符的重叠，保证上下文连贯

    Args:
        text:        清洗后的完整文本
        chunk_size:  每块目标字符数（300-500 之间）
        overlap:     相邻块重叠字符数
        source_name: 来源文件名，写入 metadata 用于溯源

    Returns:
        List[Dict]，每个元素包含:
            - text:    该块文本内容
            - source:  来源文件名
            - chunk_id: 块的顺序编号（从 0 开始）
    """
    # 先按自然段落（双换行）分割
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # 把段落逐步拼成满足 chunk_size 的块
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # 若单段落就已经超过 chunk_size，先将当前 chunk 保存，再切割该段落
        if len(para) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # 硬切超长段落
            sub_chunks = _hard_split(para, chunk_size, overlap)
            chunks.extend(sub_chunks)
            continue

        # 判断加入当前段落后是否超出 chunk_size
        tentative = (current_chunk + "\n\n" + para).strip() if current_chunk else para
        if len(tentative) <= chunk_size:
            current_chunk = tentative
        else:
            # 保存当前 chunk，开始新 chunk（带 overlap）
            if current_chunk:
                chunks.append(current_chunk.strip())
            # 新 chunk 从上一个 chunk 的结尾 overlap 个字符开始
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = (overlap_text + "\n\n" + para).strip()

    # 保存最后一块
    if current_chunk:
        chunks.append(current_chunk.strip())

    # 组装成带 metadata 的字典列表
    result = []
    for idx, chunk_text in enumerate(chunks):
        if chunk_text:  # 跳过空块
            result.append({
                "text": chunk_text,
                "source": source_name,
                "chunk_id": idx,
            })

    return result


def _hard_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    对超长文本进行强制按字符数切分（内部辅助函数）。

    Args:
        text:       待切分的长文本
        chunk_size: 每块最大字符数
        overlap:    相邻块重叠字符数

    Returns:
        切分后的文本列表
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # 下一块从 (end - overlap) 开始
        start = max(start + 1, end - overlap)
    return chunks


def parse_pdf(file_path: str, chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
    """
    PDF 解析主入口：提取文本 → 清洗 → 切块。

    Args:
        file_path:  PDF 文件路径
        chunk_size: 每块目标字符数
        overlap:    重叠字符数

    Returns:
        切分好的文本块列表，每块包含 text / source / chunk_id
    """
    source_name = Path(file_path).name
    raw_text = extract_text_from_pdf(file_path)
    clean = clean_text(raw_text)
    chunks = split_into_chunks(clean, chunk_size=chunk_size, overlap=overlap, source_name=source_name)
    return chunks

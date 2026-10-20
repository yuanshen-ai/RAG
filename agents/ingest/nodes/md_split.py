from agents.ingest.schemas import IngestGraphState, IngestGraphContext, IngestGraphStepInfo, IngestMarkdownChunk
from langgraph.runtime import Runtime
from langchain_text_splitters import MarkdownHeaderTextSplitter
from core.log import logger
from vendors.markdown_chunker import MarkdownChunkingStrategy
from agents.ainvoke_llm import ainvoke_llm_str
import asyncio
import json
import re

# 匹配 Markdown 表格的表头行 + 分隔符行模式
_PATTERN_MD_TABLE = re.compile(r"^\|.+\|$")
_PATTERN_MD_TABLE_SEP = re.compile(r"^\|(\s*[-:]+\s*\|)+$")
# 匹配 HTML 表格标签（<table>, <tr>, <td>, <th>）
_PATTERN_HTML_TABLE = re.compile(r"<(/?(?:table|tr|t[dh])\b[^>]*>)", re.IGNORECASE)
MARKDOWN_LLM_CONCURRENCY = 4


def _has_table(content: str) -> bool:
    """检查内容中是否包含表格（Markdown 表格或 HTML 表格标签）"""
    # 1. 检测 Markdown 表格：表头行 + 分隔符行模式
    lines = content.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _PATTERN_MD_TABLE.match(stripped):
            if i + 1 < len(lines) and _PATTERN_MD_TABLE_SEP.match(lines[i + 1].strip()):
                return True

    # 2. 检测 HTML 表格标签：<table>, <tr>, <td>, <th>
    if _PATTERN_HTML_TABLE.search(content):
        return True

    return False


async def _table_to_natural_language(content: str) -> str:
    """调用 LLM 将内容中的表格（Markdown 或 HTML）转换为自然语言描述，替换原表格"""
    try:
        result = await ainvoke_llm_str("table_to_text", {"content": content})
        return result
    except Exception as e:
        logger.warning(f"表格转自然语言失败，保留原始内容: {e}")
        return content


async def _process_header_chunk(
    header_chunk,
    file_name: str,
    semaphore: asyncio.Semaphore,
) -> list[IngestMarkdownChunk]:
    if not header_chunk.metadata:
        async with semaphore:
            result = await ainvoke_llm_str(
                "generate_chunk_title",
                {"content": header_chunk.page_content[:2000]},
            )
        chunk_title = {"h1": result}
    else:
        chunk_title = header_chunk.metadata
    title = json.dumps(chunk_title, ensure_ascii=False)

    content = header_chunk.page_content.replace("\r\n", "\n").replace("\r", "\n")
    strategy = MarkdownChunkingStrategy(
        min_chunk_len=200,
        soft_max_len=500,
        hard_max_len=1000,
    )
    initial_chunks = strategy.chunk_markdown(content)

    async def convert_table(initial_chunk: str) -> str:
        if not _has_table(initial_chunk):
            return initial_chunk
        async with semaphore:
            return await _table_to_natural_language(initial_chunk)

    converted_chunks = await asyncio.gather(
        *(convert_table(initial_chunk) for initial_chunk in initial_chunks)
    )
    chunks = [
        chunk
        for converted_chunk in converted_chunks
        for chunk in strategy.chunk_markdown(converted_chunk)
    ]
    return [
        IngestMarkdownChunk(
            title=title,
            content=chunk,
            header_chunk_index=index,
            file_name=file_name,
        )
        for index, chunk in enumerate(chunks)
    ]


async def md_split(state: IngestGraphState, runtime: Runtime[IngestGraphContext]):
    writer = runtime.stream_writer
    writer(IngestGraphStepInfo(name="md_split", status="running"))

    try:
        if state.markdown_content is None or state.markdown_path is None:
            raise ValueError("Markdown 内容或文件路径为空，无法切分")

        # 1. 按照标题来切分 # ## ### 
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
                ("#####", "h5"),
                ("######", "h6")
            ]
        )
        header_chunks = header_splitter.split_text(state.markdown_content)

        # 2. 不同章节及其中的表格转换受控并发处理
        semaphore = asyncio.Semaphore(MARKDOWN_LLM_CONCURRENCY)
        header_results = await asyncio.gather(
            *(
                _process_header_chunk(
                    header_chunk,
                    state.markdown_path.name,
                    semaphore,
                )
                for header_chunk in header_chunks
            )
        )
        markdown_chunks = [
            chunk
            for header_result in header_results
            for chunk in header_result
        ]
    except Exception as e:
        writer(IngestGraphStepInfo(name="md_split", status="failed", error=str(e)))
        return {"should_continue": False, "error": str(e)}
    
    writer(IngestGraphStepInfo(name="md_split", status="success"))
    logger.info(f"Markdown 切分完成，共生成 {len(markdown_chunks)} 个文本块")
    return {"markdown_chunks": markdown_chunks}

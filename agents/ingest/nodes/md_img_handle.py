import asyncio
import hashlib

from agents.ingest.schemas import IngestGraphState, IngestGraphContext, IngestGraphStepInfo
from langgraph.runtime import Runtime
import aiofiles
from clients.minio_1 import minio_client
import re
import base64
from pathlib import Path
from agents.prompts import load_prompt
from langchain_core.prompts import PromptTemplate
from langchain.messages import HumanMessage
from agents.llm import vlm_client
from core.log import logger


IMAGE_SUMMARY_CONCURRENCY = 4
IMAGE_CONTEXT_LENGTH = 100


async def image_to_base64(image_path: Path) -> str:
    async with aiofiles.open(image_path, mode="rb") as f:
        image_data = await f.read()
    b64str = base64.b64encode(image_data).decode("utf-8")
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(image_path.suffix.lower(), "application/octet-stream")
    return f"data:{mime_type};base64,{b64str}"


def _image_digest(image_path: Path) -> str:
    digest = hashlib.sha256()
    with image_path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_image_matches(md_content: str, image_name: str) -> list[re.Match[str]]:
    pattern = re.compile(
        r"!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)"
    )
    return list(pattern.finditer(md_content))


async def _summarize_image_group(
    image_paths: list[Path],
    image_digest: str,
    md_content: str,
    prompt_template: PromptTemplate,
    semaphore: asyncio.Semaphore,
) -> list[tuple[int, int, str]]:
    matches = [
        match
        for image_path in image_paths
        for match in _find_image_matches(md_content, image_path.name)
    ]
    if not matches:
        return []
    matches.sort(key=lambda match: match.start())

    image_path = image_paths[0]
    image_base64 = await image_to_base64(image_path)
    async with semaphore:
        image_url = await minio_client.upload_file(
            str(image_path),
            f"images/{image_digest}{image_path.suffix.lower()}",
        )

    first_match = matches[0]
    start, end = first_match.span()
    prompt_value = prompt_template.format(
        pre_content=md_content[max(0, start - IMAGE_CONTEXT_LENGTH):start],
        next_content=md_content[
            end:min(len(md_content), end + IMAGE_CONTEXT_LENGTH)
        ],
    )
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": prompt_value},
                {"type": "image_url", "image_url": {"url": image_base64}},
            ]
        )
    ]
    async with semaphore:
        response = await vlm_client.ainvoke(messages)

    summary = str(response.content).strip().replace("\n", "")
    replacement = f"![{summary}]({image_url})"
    return [(match.start(), match.end(), replacement) for match in matches]


async def md_img_handle(state: IngestGraphState, runtime: Runtime[IngestGraphContext]):
    writer = runtime.stream_writer
    writer(IngestGraphStepInfo(name="md_img_handle", status="running"))

    if state.markdown_path is None:
        writer(IngestGraphStepInfo(name="md_img_handle", status="failed", error="markdown_path is None"))
        return {"should_continue": False, "error": "markdown_path is None"}
    
    try:
        async with aiofiles.open(state.markdown_path, 'r', encoding='utf-8') as f:
            md_content = await f.read()
        
        image_dir = state.markdown_dir / "images"
        if not image_dir.exists():
            writer(IngestGraphStepInfo(name="md_img_handle", status="success"))
            return {"markdown_content": md_content}

        image_paths = [path for path in image_dir.glob("*") if path.is_file()]
        referenced_paths = [
            path
            for path in image_paths
            if _find_image_matches(md_content, path.name)
        ]
        if not referenced_paths:
            writer(IngestGraphStepInfo(name="md_img_handle", status="success"))
            return {"markdown_content": md_content}

        digest_values = await asyncio.gather(
            *(asyncio.to_thread(_image_digest, path) for path in referenced_paths)
        )
        image_groups: dict[str, list[Path]] = {}
        for image_path, digest in zip(referenced_paths, digest_values):
            image_groups.setdefault(digest, []).append(image_path)

        # 加载提示词模板
        prompt = await load_prompt("summarize_image")
        prompt_template = PromptTemplate(
            template=prompt,
            input_variables=["pre_content", "next_content"]
        )

        semaphore = asyncio.Semaphore(IMAGE_SUMMARY_CONCURRENCY)
        results = await asyncio.gather(
            *(
                _summarize_image_group(
                    paths,
                    digest,
                    md_content,
                    prompt_template,
                    semaphore,
                )
                for digest, paths in image_groups.items()
            ),
            return_exceptions=True,
        )

        replacements: list[tuple[int, int, str]] = []
        for digest, result in zip(image_groups, results):
            if isinstance(result, BaseException):
                logger.warning(f"图片摘要失败，保留原始引用: {digest} - {result}")
                continue
            replacements.extend(result)

        for start, end, replacement in sorted(replacements, reverse=True):
            md_content = md_content[:start] + replacement + md_content[end:]
            
        # 3. 将处理后的markdown内容写回文件
        # 实际开发中不需要写入这个文件的
        # async with aiofiles.open(state.markdown_dir / "new_md.md", 'w', encoding='utf-8') as f:
        #     await f.write(md_content)
    except Exception as e:
        writer(IngestGraphStepInfo(name="md_img_handle", status="failed", error=str(e)))
        return {"should_continue": False, "error": str(e)}
        
    writer(IngestGraphStepInfo(name="md_img_handle", status="success"))
    return {"markdown_content": md_content}

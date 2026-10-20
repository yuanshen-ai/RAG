from fastapi import APIRouter, UploadFile, File
import asyncio
import shutil
import uuid
import tempfile
from pathlib import Path
import aiofiles
from clients.minio_1 import minio_client
from core.log import logger
from fastapi.responses import StreamingResponse
from agents.ingest.graph import graph
from agents.ingest.schemas import IngestGraphState, IngestGraphContext, IngestGraphStepInfo
from repositories.milvus import MilvusEntityRepository, MilvusChunkRepository
from clients.milvus import milvus_client
import json

router = APIRouter()

async def _event_stream(
    original_name: str,
    local_path: Path,
    file_url: str,
    object_name: str,
    markdown_dir: Path
):
    """SSE事件生成器"""
    try:
        state = IngestGraphState(
            file_url=file_url,
            file_path=local_path,
            markdown_dir=markdown_dir
        )
        context = IngestGraphContext(
            milvus_chunk_repo=MilvusChunkRepository(milvus_client.client),
            milvus_entity_repo=MilvusEntityRepository(milvus_client.client)
        )
        async for chunk in graph.astream(state, context=context, stream_mode="custom"):
            if isinstance(chunk, IngestGraphStepInfo):
                yield f"event: step\ndata: {chunk.model_dump_json(ensure_ascii=False)}\n\n"
        # 发送完成事件
        yield f"event: done\ndata: {json.dumps({'file_name': original_name, 'object_name': object_name}, ensure_ascii=False)}\n\n"
    except Exception as e:
        logger.error(f"执行提取文档agent失败：{str(e)}")
        yield f"event: error\ndata: {json.dumps({'message': str(e)}, ensure_ascii=False)}"
    finally:
        await asyncio.to_thread(shutil.rmtree, local_path.parent, ignore_errors=True)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    上传pdf文件
    """
    # 1. 生成文件的object name
    original_name = file.filename or "unknown.pdf"
    file_id = uuid.uuid4().hex
    object_name = f"pdfs/{file_id}/{original_name}"

    # 2. 将pdf文件存储到服务器（临时文件）+上传到MinIO
    # Windows: C:\Users\用户名\AppData\Local\Temp
    # Linux：/tmp
    temp_dir = Path(tempfile.mkdtemp())
    local_path = temp_dir / original_name
    markdown_dir = temp_dir / "markdown"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    try:
        async with aiofiles.open(local_path, "wb") as fp:
            while content := await file.read(1024 * 1024):
                await fp.write(content)

        # 上传到MinIO
        file_url = await minio_client.upload_file(str(local_path.resolve()), object_name)
        logger.info(f"上传文件url：{file_url}")
    except Exception as e:
        logger.info(f"文件上传失败：{str(e)}")
        raise

    # 流式返回ingest agent执行的情况
    # 一般情况下，使用StreamingResponse返回数据，都是通过一个函数来yield
    return StreamingResponse(
        _event_stream(original_name, local_path, file_url, object_name, markdown_dir),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

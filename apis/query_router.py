from fastapi import APIRouter
from pydantic import BaseModel, Field
import uuid
from fastapi.responses import StreamingResponse
from agents.query.graph import get_graph
from agents.query.schemas import QueryGraphState, QueryGraphContext
from clients.milvus import milvus_client
from repositories.milvus import MilvusChunkRepository, MilvusEntityRepository
from core.log import logger
from langchain.messages import HumanMessage
import json


router = APIRouter()

class QueryData(BaseModel):
    question: str = Field(..., description="用户的提问内容",min_length=1)
    thread_id: str | None = Field(None, description="会话ID，不传则创建新的")


async def _event_stream(question: str, thread_id: str):
    state = QueryGraphState(
        messages=[HumanMessage(content=question)],
        query=question
    )
    context = QueryGraphContext(
        milvus_chunk_repo=MilvusChunkRepository(milvus_client.client),
        milvus_entity_repo=MilvusEntityRepository(milvus_client.client)
    )

    graph = await get_graph()
    try:
        async for chunk in graph.astream(
            state, 
            context=context, #type: ignore
            stream_mode="custom",
            config={"thread_id": thread_id}
        ):
            chunk_type = chunk.get('type')
            chunk_data = chunk.get("data")
            yield f"event: {chunk_type}\ndata: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
        # 发送完成事件
        yield f"event: done\ndata: {{}}\n\n"
    except Exception as e:
        logger.error(f"查询失败：{str(e)}")
        yield f"event: error\ndata: {json.dumps({'message': str(e)})}"
    


@router.post("")
async def query(data: QueryData):
    thread_id = data.thread_id or uuid.uuid4().hex
    return StreamingResponse(
        _event_stream(data.question, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

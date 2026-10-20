import dashscope
from dtos.rerank import ReRankChunk, ReRankedChunk
from http import HTTPStatus
from asgiref.sync import sync_to_async
from conf import app_config


async def rerank_documents(query: str, chunks: list[ReRankChunk]) -> list[ReRankedChunk] | None:
    assert len(chunks) > 0
    response = await sync_to_async(dashscope.TextReRank.call)(
        model="qwen3-rerank",
        query=query,
        documents=[chunk.content for chunk in chunks],
        return_documents=True,
        api_key=app_config.bailian.api_key
    )
    if response.status_code == HTTPStatus.OK:
        results = response['output']['results']
        reranked_chunks: list[ReRankedChunk] = []
        for result in results:
            index = result["index"]
            score = result['relevance_score']
            rerank_chunk = chunks[index]
            reranked_chunks.append(ReRankedChunk(score=score, **rerank_chunk.model_dump()))
        return reranked_chunks
    else:
        raise RuntimeError(response.message)

if __name__ == "__main__":
    import asyncio
    chunks: list[ReRankChunk] = [
        ReRankChunk(title="什么是重排序模型", content="重排序模型广泛应用于搜索引擎和推荐系统，按相关性对候选文本进行排序", source="embedding"),
        ReRankChunk(title="什么是重排序模型", content="量子计算是计算科学的前沿领域", source="embedding"),
        ReRankChunk(title="什么是重排序模型", content="预训练语言模型的发展为重排序模型带来了新的进展", source="embedding"),
    ]
    async def test():
        await rerank_chunks(chunks)
    asyncio.run(test())
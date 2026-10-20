from langgraph.graph import StateGraph, START, END
from agents.query.schemas import QueryGraphState, QueryGraphContext
from agents.query.nodes.intention_parse import intention_parse
from agents.query.nodes.entity_confirm import entity_confirm
from agents.query.nodes.embedding_search import embedding_search
from agents.query.nodes.hyde_search import hyde_search
from agents.query.nodes.rrf_merge import rrf_merge
from agents.query.nodes.web_search import web_search
from agents.query.nodes.chunks_rerank import chunks_rerank
from agents.query.nodes.result_fusion import result_fusion
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from clients.postgre import postgre_client
from clients.milvus import milvus_client
from repositories.milvus import MilvusChunkRepository, MilvusEntityRepository
from core.log import logger
from pydantic import BaseModel
from core.asyncio_compat import run_async


def build_graph_builder() -> StateGraph:
    graph_builder = StateGraph(
        state_schema=QueryGraphState, 
        context_schema=QueryGraphContext
    )
    graph_builder.add_node(intention_parse)
    graph_builder.add_node(entity_confirm)
    graph_builder.add_node(embedding_search)
    graph_builder.add_node(hyde_search)
    graph_builder.add_node(rrf_merge)
    graph_builder.add_node(web_search)
    graph_builder.add_node(chunks_rerank)
    graph_builder.add_node(result_fusion)

    graph_builder.add_edge(START, "intention_parse")

    graph_builder.add_conditional_edges(
        "intention_parse", 
        lambda state: "entity_confirm" if state.should_continue else END
    )

    graph_builder.add_edge(
        "entity_confirm", 
        "embedding_search"
    )
    graph_builder.add_edge(
        "entity_confirm",
        "hyde_search"
    )
    graph_builder.add_edge(
        "entity_confirm", 
        "web_search"
    )

    # Wait for both retrieval branches before merging their RRF scores.
    graph_builder.add_edge(["embedding_search", "hyde_search", "web_search"], "rrf_merge")

    graph_builder.add_edge("rrf_merge", "chunks_rerank")

    graph_builder.add_edge("chunks_rerank", "result_fusion")
    graph_builder.add_edge("result_fusion", END)

    return graph_builder #type: ignore

async def get_graph():
    graph_builder = build_graph_builder()
    # 注意，这里没有对postgre_client进行init操作，后续集成到FastAPI项目中要记得初始化
    checkpointer = AsyncPostgresSaver(conn=postgre_client.pool)
    await checkpointer.setup()
    graph = graph_builder.compile(checkpointer=checkpointer)
    return graph

if __name__ == "__main__":
    from langchain.messages import HumanMessage

    async def main():
        await postgre_client.init()
        graph = await get_graph()
        query = "请问 HAK 180 供电数据是什么？比如电流、电压"
        # query = "你好"
        state = QueryGraphState(
            messages=[HumanMessage(content=query)],
            query=query
        )
        milvus_entity_repo = MilvusEntityRepository(milvus_client.client)
        milvus_chunk_repo = MilvusChunkRepository(milvus_client.client)
        context = QueryGraphContext(
            milvus_entity_repo=milvus_entity_repo,
            milvus_chunk_repo=milvus_chunk_repo
        )
        async for chunk in graph.astream(
            state, 
            context=context, #type: ignore
            stream_mode="custom",
            config={"thread_id": "1"}
        ):
            print(chunk)
        await postgre_client.close()
    
    run_async(main())
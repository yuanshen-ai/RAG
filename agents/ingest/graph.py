from langgraph.graph import StateGraph, START, END
from agents.ingest.schemas import IngestGraphState, IngestGraphContext
from agents.ingest.nodes.pdf_to_md import pdf_to_md
from agents.ingest.nodes.md_split import md_split
from agents.ingest.nodes.md_img_handle import md_img_handle
from agents.ingest.nodes.entity_recognize import entity_recognize
from agents.ingest.nodes.text_to_embedding import text_to_embedding
from repositories.milvus import MilvusEntityRepository, MilvusChunkRepository
from clients.milvus import milvus_client


graph_builder = StateGraph(
    state_schema=IngestGraphState,
    context_schema=IngestGraphContext
)

graph_builder.add_node(pdf_to_md)
graph_builder.add_node(md_img_handle)
graph_builder.add_node(md_split)
graph_builder.add_node(entity_recognize)
graph_builder.add_node(text_to_embedding)

graph_builder.add_edge(START, "pdf_to_md")
graph_builder.add_conditional_edges(
    "pdf_to_md", 
    lambda state: "md_img_handle" if state.should_continue else END
)
graph_builder.add_conditional_edges(
    "md_img_handle", 
    lambda state: "md_split" if state.should_continue else END
)
graph_builder.add_conditional_edges(
    "md_split", 
    lambda state: "entity_recognize" if state.should_continue else END
)
graph_builder.add_conditional_edges(
    "entity_recognize", 
    lambda state: "text_to_embedding" if state.should_continue else END
)
graph_builder.add_edge("text_to_embedding", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    import asyncio
    import shutil
    import tempfile
    import uuid
    from pathlib import Path

    from clients.minio_1 import minio_client

    async def ingest_pdf(pdf_path: Path, context: IngestGraphContext) -> bool:
        """Upload and ingest one PDF in an isolated temporary directory."""
        with tempfile.TemporaryDirectory(prefix="rag_ingest_") as temp_dir:
            work_dir = Path(temp_dir)
            local_path = work_dir / pdf_path.name
            markdown_dir = work_dir / "markdown"

            await asyncio.to_thread(shutil.copy2, pdf_path, local_path)

            object_name = f"pdfs/{uuid.uuid4().hex}/{pdf_path.name}"
            file_url = await minio_client.upload_file(
                str(local_path.resolve()),
                object_name,
            )

            state = IngestGraphState(
                file_url=file_url,
                file_path=local_path,
                markdown_dir=markdown_dir,
            )

            succeeded = True
            async for chunk in graph.astream(
                state,
                context=context,
                stream_mode="custom",
            ):
                print(chunk)
                if getattr(chunk, "status", None) == "failed":
                    succeeded = False

            return succeeded

    async def main():
        project_dir = Path(__file__).resolve().parents[2]
        uploads_dir = project_dir / "uploads"
        pdf_paths = sorted(
            path for path in uploads_dir.glob("*.pdf") if path.is_file()
        )

        if not pdf_paths:
            print(f"未在 {uploads_dir} 中找到 PDF 文件")
            return

        milvus_entity_repo = MilvusEntityRepository(milvus_client.client)
        milvus_chunk_repo = MilvusChunkRepository(milvus_client.client)
        await milvus_entity_repo.ensure_collection()
        await milvus_chunk_repo.ensure_collection()
        context = IngestGraphContext(
            milvus_entity_repo=milvus_entity_repo,
            milvus_chunk_repo=milvus_chunk_repo,
        )

        success_count = 0
        for index, pdf_path in enumerate(pdf_paths, start=1):
            print(f"[{index}/{len(pdf_paths)}] 开始入库: {pdf_path.name}")
            try:
                if await ingest_pdf(pdf_path, context):
                    success_count += 1
                    print(f"[{index}/{len(pdf_paths)}] 入库完成: {pdf_path.name}")
                else:
                    print(f"[{index}/{len(pdf_paths)}] 入库失败，已跳过: {pdf_path.name}")
            except Exception as exc:
                print(
                    f"[{index}/{len(pdf_paths)}] 入库异常，已跳过: "
                    f"{pdf_path.name} - {exc}"
                )

        print(f"处理结束：成功 {success_count} 本，失败 {len(pdf_paths) - success_count} 本")

    asyncio.run(main())

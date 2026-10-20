from agents.ingest.schemas import IngestGraphState, IngestGraphContext, IngestGraphStepInfo
from langgraph.runtime import Runtime
from integrations.mineru import mineru_integration

async def pdf_to_md(state: IngestGraphState, runtime: Runtime[IngestGraphContext]):
    writer = runtime.stream_writer
    writer(IngestGraphStepInfo(name="pdf_to_md", status="running"))

    try:
        markdown_path = await mineru_integration.parse_pdf(
            state.file_path,
            state.markdown_dir,
        )
        writer(IngestGraphStepInfo(name="pdf_to_md", status="success"))
        return {"markdown_path": markdown_path}
    except Exception as e:
        writer(IngestGraphStepInfo(name="pdf_to_md", status="failed", error=str(e)))
        return {"should_continue": False, "error": str(e)}


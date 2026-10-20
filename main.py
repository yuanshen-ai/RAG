from fastapi import FastAPI
from apis.ingest_router import router as ingest_router
from apis.query_router import router as query_router
import uvicorn
from contextlib import asynccontextmanager
from clients.postgre import postgre_client
from repositories.milvus import MilvusEntityRepository, MilvusChunkRepository
from clients.milvus import milvus_client
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
from fastapi.staticfiles import StaticFiles
#cd docker  docker compose up -d  docker ps

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    await postgre_client.init()
    chunk_repo = MilvusChunkRepository(milvus_client.client)
    entity_repo = MilvusEntityRepository(milvus_client.client)
    await chunk_repo.ensure_collection()
    await entity_repo.ensure_collection()
    yield
    await postgre_client.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

app.include_router(ingest_router, prefix="/api/ingest", tags=['文档提取'])
app.include_router(query_router, prefix="/api/query", tags=['搜索'])

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = TEMPLATE_DIR / "ingest.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.get("/query", response_class=HTMLResponse)
async def query():
    html_path = TEMPLATE_DIR / "query.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, loop="asyncio")

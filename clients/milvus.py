from pymilvus import AsyncMilvusClient
from conf import app_config


class MilvusClient:
    def __init__(self) -> None:
        uri = f"http://{app_config.milvus.host}:{app_config.milvus.port}"
        self.client = AsyncMilvusClient(
            uri=uri,
            token=f"{app_config.milvus.user}:{app_config.milvus.password}"
        )
    async def close(self):
        await self.client.close()


milvus_client = MilvusClient()
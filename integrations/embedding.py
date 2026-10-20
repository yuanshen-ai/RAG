import dashscope
from conf import app_config
from typing import Literal
from asgiref.sync import sync_to_async
from core.log import logger


async def generate_texts_embeddings(
    texts: list[str], 
    output_type: Literal['dense', 'sparse', 'dense&sparse'] = "dense&sparse"
) -> list[dict[str, list[float] | dict[int, float]]] | None:
    try:
        response = await sync_to_async(
            dashscope.TextEmbedding.call,
            thread_sensitive=False,
        )(
            model="text-embedding-v4",
            input=texts,
            api_key=app_config.bailian.api_key,
            output_type=output_type
        )
        embeddings = response['output']['embeddings']
        result = []
        for embedding in embeddings:
            dense_embeddings = embedding.get('embedding')
            raw_sparse_embeddings = embedding.get('sparse_embedding')
            if raw_sparse_embeddings:
                sparse_embeddings = {e['index']: e['value'] for e in raw_sparse_embeddings}

            if output_type == "dense":
                result.append({"dense": dense_embeddings})
            elif output_type == "sparse":
                result.append({"dense": sparse_embeddings})
            else:
                assert dense_embeddings is not None
                assert sparse_embeddings is not None
                result.append({"dense": dense_embeddings, "sparse": sparse_embeddings})
        return result
    except Exception as e:
        logger.error(str(e))
        return None

import json

from minio import Minio
from conf import app_config
from core.log import logger
from asgiref.sync import sync_to_async


class RAGMinioClient:
    def __init__(self):
        self.client = Minio(
            endpoint=app_config.minio.endpoint,
            access_key=app_config.minio.access_key,
            secret_key=app_config.minio.secret_key,
            secure=app_config.minio.secure
        )
        self.bucket_name = app_config.minio.bucket_name

        # 确保 bucket 存在
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)
            logger.info(f"Bucket '{self.bucket_name}' created.")
        else:
            logger.info(f"Bucket '{self.bucket_name}' already exists.")

        # 配置存储桶公网只读策略：允许匿名用户通过URL直接访问桶内文件
        bucket_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},  # *表示所有匿名用户（S3兼容标识）
                "Action": ["s3:GetObject"],   # 仅授权文件获取/访问操作
                "Resource": [f"arn:aws:s3:::{self.bucket_name}/*"]
            }]
        }
        self.client.set_bucket_policy(self.bucket_name, json.dumps(bucket_policy))

    async def upload_file(self, file_path: str, object_name: str) -> str:
        #nafput_object = sync_to_async(self.client.fput_object)
        # await afput_object(self.bucket_name, object_name, file_path)
        await sync_to_async(
            self.client.fput_object,
            thread_sensitive=False,
        )(self.bucket_name, object_name, file_path)
        return f"http://{app_config.minio.endpoint}/{self.bucket_name}/{object_name}"

    async def close(self):
        # Minio Python SDK没有提供显式的close方法，但如果有需要，可以在这里执行一些清理操作
        pass


minio_client = RAGMinioClient()

if __name__ == "__main__":
    import asyncio
    async def test():
        file_url = await minio_client.upload_file("main.py", "main.py")
        print(f"File uploaded to: {file_url}")
    asyncio.run(test())

# 上传副本说明

本目录未携带原 Git 历史、本地依赖、日志、上传文档或 PDF 解析结果。

启动前：
1. 将 conf/app_config.example.yaml 复制为 conf/app_config.yaml，并填写自己的服务配置。
2. 将 docker/.env.example 复制为 docker/.env，并填写自己的 PostgreSQL 与 MinIO 凭据。
3. 将 docker/milvus/milvus.example.yaml 复制为 docker/milvus/milvus.yaml，填写自己的凭据。
4. 保证应用配置、Docker 环境变量、Milvus 的 MinIO accessKeyID/secretAccessKey 一致。Milvus root 密码与应用 milvus.password 一致；PostgreSQL 密码与 docker/.env 一致。
5. 自行安装依赖和启动所需服务；重新上传需要检索的文档。

实际配置和业务数据被 .gitignore 排除。此目录尚未初始化 Git，也未上传 GitHub。

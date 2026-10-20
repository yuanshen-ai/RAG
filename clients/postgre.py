from typing import cast
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from conf import app_config

class RagPostgreClient:
    """使用 psycopg 异步连接池管理 PostgreSQL 连接的客户端类"""

    def __init__(self) -> None:
        """初始化客户端，读取配置并拼接数据库连接 URI"""
        self._pool = None  # 初始化连接池对象为空
        
        # 从应用配置中获取 PostgreSQL 的连接参数
        db_user = app_config.postgre.user
        db_password = app_config.postgre.password
        db_host = app_config.postgre.host
        db_port = app_config.postgre.port
        db_name = app_config.postgre.db_name
        
        # 构建标准的 PostgreSQL 连接 URI 字符串
        self._uri = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    async def init(self):
        """
        异步初始化连接池
        创建连接池实例，打开连接并等待连接就绪
        """
        if self._pool is None:
            # 使用 cast 进行类型注解，声明连接池中的连接会返回字典类型的行数据 (DictRow)
            self._pool = cast(
                AsyncConnectionPool[AsyncConnection[DictRow]],
                AsyncConnectionPool(
                    self._uri,
                    min_size=1,       # 连接池保持的最小连接数
                    max_size=10,      # 连接池允许的最大连接数
                    open=False,       # 延迟打开连接池，通过后续显式调用 .open() 打开
                    kwargs={
                        "autocommit": True,       # 开启自动提交模式（执行 SQL 后自动提交事务）
                        "row_factory": dict_row,  # 配置行工厂，使查询结果以字典(dict)形式返回，而非元组(tuple)
                        "prepare_threshold": 0,   # 禁用预处理语句(Prepared Statements)的自动生成
                    },
                ),
            )
            await self._pool.open()  # 异步打开连接池
            await self._pool.wait()  # Wait 确保连接池中至少有一个连接已成功建立并就绪
        return self

    @property
    def pool(self):
        """
        获取连接池实例的属性方法
        如果在未初始化时调用，会抛出运行时错误
        """
        if not self._pool:
            raise RuntimeError("PostgreSQL client not inited. Call init() first.")
        return self._pool

    async def close(self):
        """异步关闭连接池，释放所有数据库连接"""
        if self._pool:
            await self._pool.close()  # 异步关闭连接池
            self._pool = None         # 将连接池对象重置为 None

# 实例化全局的 PostgreSQL 客户端对象，供应用其他地方导入和使用
postgre_client = RagPostgreClient()
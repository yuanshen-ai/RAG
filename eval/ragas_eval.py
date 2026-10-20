"""
RAGAS 评估脚本

对 eval/dataset/ 下所有 CSV 文件中的每条 QA 数据，先运行 RAG query pipeline 获取检索结果和答案，
然后使用 RAGAS 计算以下 5 项指标：
- Context Relevance（上下文相关性）：检索到的上下文与问题的相关程度
- Context Recall（上下文召回率）：参考答案中的信息能否从检索上下文中找到
- Context Precision（上下文精确度）：检索到的上下文是否有助于回答问题
- Faithfulness（忠实度）：生成的答案是否完全基于检索上下文（无幻觉）
- Answer Relevance（答案相关性）：生成的答案与问题的相关程度

评估结果打印到控制台并保存到 eval/ragas_eval_result.json。

用法：
    uv run python -m eval.ragas_eval
    uv run python -m eval.ragas_eval -i eval/dataset/ -o eval/result.json
    uv run python -m eval.ragas_eval -i eval/dataset/xxx.csv -o eval/result.json
或：
    uv run python eval/ragas_eval.py -i eval/dataset/ -o eval/result.json
"""

import argparse
import asyncio
import csv
import json
import sys
import types
from pathlib import Path

# ============================================================================
# 修复 ragas 导入兼容性问题
# ragas 0.4.3 依赖旧版 langchain-community 的 vertexai 模块，
# 但新版 langchain-community 0.4.x 已将其移除，这里做 monkey-patch
# ============================================================================
_dummy_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
_dummy_vertexai.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore
_dummy_vertexai.VertexAI = type("VertexAI", (), {})  # type: ignore
sys.modules["langchain_community.chat_models.vertexai"] = _dummy_vertexai

import numpy as np
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.metrics.collections import (
    ContextRelevance,
    ContextRecall,
    ContextPrecisionWithReference,
    Faithfulness,
)
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
from langchain_core.messages import HumanMessage

from conf import app_config
from agents.query.graph import build_graph_builder
from agents.query.schemas import QueryGraphState, QueryGraphContext
from repositories.milvus import MilvusChunkRepository, MilvusEntityRepository
from clients.milvus import milvus_client
from integrations.embedding import generate_texts_embeddings
from core.asyncio_compat import run_async
from core.log import logger

# ============================================================================
# 评估专用日志：错误日志写入 eval/eval.log
# ============================================================================
_EVAL_LOG_DIR = Path(__file__).parent
_EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
_eval_logger_id = logger.add(
    sink=_EVAL_LOG_DIR / "eval.log",
    level="WARNING",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
)

# ============================================================================
# 默认配置
# ============================================================================
DEFAULT_INPUT = Path(__file__).parent / "dataset"       # 默认输入目录
DEFAULT_OUTPUT = Path(__file__).parent / "ragas_eval_result.json"  # 默认输出文件


# ============================================================================
# RAGAS Embedding 封装
# ============================================================================
class DashScopeEmbedding(BaseRagasEmbedding):
    """将项目内的 DashScope text-embedding-v4 封装为 RAGAS 兼容的 Embedding。

    注意：百炼的 compatible-mode 端点仅支持 Chat Completions，
    不支持 /v1/embeddings 接口，因此不能直接使用 OpenAIEmbeddings。
    这里直接调用项目已有的 dashscope embedding 函数。
    """

    def embed_text(self, text: str, **kwargs) -> list[float]:
        """同步嵌入，内部通过 asyncio.run 调用异步函数"""
        return asyncio.run(self.aembed_text(text, **kwargs))

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        """异步嵌入单个文本"""
        result = await generate_texts_embeddings([text], output_type="dense")
        if result is None or len(result) == 0:
            raise RuntimeError(f"文本嵌入失败: {text[:50]}...")
        return result[0]["dense"]  # type: ignore[return-value]


# ============================================================================
# 数据加载
# ============================================================================
def load_qa_data(input_path: Path) -> list[dict[str, str]]:
    """从指定的 CSV 文件或目录加载 QA 评测数据。

    - 如果是目录：扫描目录下所有 .csv 文件，合并加载
    - 如果是文件：直接加载该 CSV 文件

    每个 CSV 文件必须包含 question 和 ground_truth 两列，编码为 UTF-8。
    返回合并后的所有 QA 记录列表。
    """
    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"在 {input_path} 下未找到任何 CSV 文件")
    elif input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"不支持的文件类型: {input_path.suffix}，仅支持 .csv 文件")
        csv_files = [input_path]
    else:
        raise FileNotFoundError(f"路径不存在: {input_path}")

    rows: list[dict[str, str]] = []
    for csv_path in csv_files:
        file_rows = 0
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row.get("question", "").strip()
                ground_truth = row.get("ground_truth", "").strip()
                if question and ground_truth:
                    rows.append({
                        "question": question,
                        "ground_truth": ground_truth,
                        "source_file": csv_path.name,  # 记录来源文件，便于追溯
                    })
                    file_rows += 1
        print(f"   📄 {csv_path.name}: {file_rows} 条")

    return rows


# ============================================================================
# Query Pipeline 执行
# ============================================================================
async def run_query_pipeline(
    graph, question: str
) -> tuple[list[str] | None, str | None, str | None]:
    """运行 RAG query pipeline，返回 (检索上下文列表, 答案文本, 错误信息)。

    - 检索上下文：取 reranked_chunks 中各 chunk 的 content
    - 答案文本：取最终 AIMessage 的 content
    """
    state = QueryGraphState(
        messages=[HumanMessage(content=question)],
        query=question,
    )
    entity_repo = MilvusEntityRepository(milvus_client.client)
    chunk_repo = MilvusChunkRepository(milvus_client.client)
    context_obj = QueryGraphContext(
        milvus_entity_repo=entity_repo,
        milvus_chunk_repo=chunk_repo,
    )

    try:
        result = await graph.ainvoke(state, context=context_obj)
    except Exception as exc:
        logger.error(f"Graph invoke 失败 (question='{question[:30]}...'): {exc}")
        return None, None, str(exc)

    # 提取检索上下文（重排序后的 chunk 内容）
    reranked_chunks = result.get("reranked_chunks")
    contexts: list[str] | None = None
    if reranked_chunks:
        contexts = [chunk.content for chunk in reranked_chunks]

    # 提取最终答案（最后一条 AI 消息）
    messages = result.get("messages", [])
    answer: str | None = None
    for msg in reversed(messages):
        if hasattr(msg, "content") and getattr(msg, "type", None) == "ai":
            answer = msg.content
            break

    error = result.get("error")
    return contexts, answer, error


# ============================================================================
# RAGAS 组件构建
# ============================================================================
def build_ragas_llm():
    """基于项目百炼配置构建 RAGAS 评估用的 LLM。

    使用 AsyncOpenAI 客户端指向百炼的 OpenAI-compatible 端点，
    模型为项目配置中的 qwen3.5-flash。
    """
    client = AsyncOpenAI(
        base_url=app_config.llm.base_url.rstrip("/"),
        api_key=app_config.llm.api_key,
        timeout=120.0,  # 长答案的 Faithfulness 评估需要较长时间
        max_retries=2,  # SDK 层面最多重试 2 次
    )
    return llm_factory(
        model=app_config.llm.model_name,
        client=client,
        max_tokens=8192,  # Faithfulness 等指标需要更长的输出
    )


def build_metrics(llm, embeddings: DashScopeEmbedding) -> list:
    """构建全部 5 项 RAGAS 评估指标"""
    return [
        ContextRelevance(llm=llm),
        ContextRecall(llm=llm),
        ContextPrecisionWithReference(llm=llm),
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
    ]


# ============================================================================
# 单条评估
# ============================================================================
async def score_single_sample(
    metrics: list,
    question: str,
    ground_truth: str,
    contexts: list[str] | None,
    answer: str | None,
) -> tuple[dict[str, float | None], dict[str, str | None]]:
    """对单条样本执行所有指标的评分。

    返回 (scores, errors)，errors 记录各指标失败原因，成功时为 None。
    """
    scores: dict[str, float | None] = {}
    errors: dict[str, str | None] = {}
    for metric in metrics:
        metric_name = metric.name
        try:
            result = await _dispatch_metric(metric_name, metric, question, ground_truth, contexts, answer)
            # 从 MetricResult 中提取浮点数；value 可能是 np.nan（LLM 调用失败时）
            if result is None:
                scores[metric_name] = None
                errors[metric_name] = "缺少必要的输入（contexts 或 answer 为空）"
            else:
                val = float(result.value)
                if not np.isnan(val):
                    scores[metric_name] = val
                    errors[metric_name] = None
                else:
                    scores[metric_name] = None
                    errors[metric_name] = "评分结果为 NaN（LLM 调用可能失败）"
        except TypeError as exc:
            # ragas 0.4.3 在 LLM 返回异常值时可能出现 "float has no len()" 等 TypeErrors
            msg = f"TypeError: {exc}"
            scores[metric_name] = None
            errors[metric_name] = msg
            logger.warning(f"[{metric_name}] 评分失败（类型错误）: {exc} | question={question[:50]}...")
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            scores[metric_name] = None
            errors[metric_name] = msg
            logger.warning(f"[{metric_name}] 评分失败: {exc} | question={question[:50]}...")
    return scores, errors


async def _dispatch_metric(metric_name: str, metric, question: str, ground_truth: str, contexts: list[str] | None, answer: str | None):
    """根据指标名称调用对应的 ascore 方法"""
    if metric_name == "context_relevance":
        if not contexts:
            return None
        return await metric.ascore(user_input=question, retrieved_contexts=contexts)

    elif metric_name == "context_recall":
        if not contexts:
            return None
        return await metric.ascore(user_input=question, retrieved_contexts=contexts, reference=ground_truth)

    elif metric_name == "context_precision_with_reference":
        if not contexts:
            return None
        return await metric.ascore(user_input=question, reference=ground_truth, retrieved_contexts=contexts)

    elif metric_name == "faithfulness":
        if not contexts or not answer:
            return None
        return await metric.ascore(user_input=question, response=answer, retrieved_contexts=contexts)

    elif metric_name == "answer_relevancy":
        if not answer:
            return None
        return await metric.ascore(user_input=question, response=answer)

    return None


# ============================================================================
# 主流程
# ============================================================================
async def evaluate_all(input_path: Path, output_path: Path):
    """主评估流程：加载数据 → 构建 pipeline → 逐条顺序评估 → 汇总输出

    采用异步调用（ainvoke / ascore），但逐条顺序执行，不并发。
    """
    # --- 1. 加载数据 ---
    path_type = "目录" if input_path.is_dir() else "文件"
    print(f"📂 扫描输入{path_type}: {input_path}")
    qa_pairs = load_qa_data(input_path)
    print(f"📄 共加载 {len(qa_pairs)} 条 QA 数据\n")

    # --- 2. 构建 graph（无 checkpoint，评估场景不需要多轮对话持久化） ---
    print("🔧 编译 Query Graph（无 checkpoint）...")
    graph_builder = build_graph_builder()
    graph = graph_builder.compile()

    # --- 3. 构建 RAGAS 组件 ---
    print("🔧 构建 RAGAS 评估组件...")
    ragas_llm = build_ragas_llm()
    ragas_embeddings = DashScopeEmbedding()
    metrics = build_metrics(ragas_llm, ragas_embeddings)
    metric_names = [m.name for m in metrics]
    print(f"   指标: {metric_names}\n")

    # --- 4. 逐条顺序评估（异步调用，但不并发，一条一条执行） ---
    all_results: list[dict] = []
    total = len(qa_pairs)

    for idx, qa in enumerate(qa_pairs):
        question = qa["question"]
        ground_truth = qa["ground_truth"]

        # 运行 query pipeline（异步）
        contexts, answer, pipeline_error = await run_query_pipeline(graph, question)

        if pipeline_error:
            scores = {}
            score_errors = {}
        else:
            # 逐指标评分（异步）
            scores, score_errors = await score_single_sample(metrics, question, ground_truth, contexts, answer)

        ctx_count = len(contexts) if contexts else 0

        # 打印结果
        completed = idx + 1
        print(f"{'─' * 60}")
        print(f"[{completed}/{total}] {question[:70]}...")
        if pipeline_error:
            print(f"  ⚠️  Pipeline 出错: {pipeline_error}")
        else:
            print(f"  检索上下文: {ctx_count} 条 | 答案长度: {len(answer) if answer else 0} 字符")
            if not contexts:
                print(f"  ⚠️  未检索到上下文，跳过需要上下文的指标")
            if not answer:
                print(f"  ⚠️  未生成答案，跳过需要答案的指标")
            for name, val in scores.items():
                if val is not None:
                    print(f"  {name}: {val:.4f}")
                else:
                    err_msg = score_errors.get(name, "未知错误")
                    print(f"  {name}: N/A（{err_msg}）")

        all_results.append({
            "question": question,
            "ground_truth": ground_truth,
            "source_file": qa.get("source_file", ""),
            "answer": answer,
            "contexts_count": ctx_count,
            "pipeline_error": pipeline_error,
            "scores": scores,
            "score_errors": score_errors,
        })

    # --- 5. 汇总统计 ---
    print(f"\n{'=' * 60}")
    print("📊 汇总统计")
    print(f"{'=' * 60}")

    summary: dict[str, dict] = {}
    for name in metric_names:
        values = [
            r["scores"].get(name)
            for r in all_results
            if r["scores"].get(name) is not None
        ]
        if values:
            avg = sum(values) / len(values)
            summary[name] = {
                "count": len(values),
                "mean": round(avg, 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }
            print(f"  {name}:")
            print(f"    均值={avg:.4f}  最小={min(values):.4f}  最大={max(values):.4f}  有效={len(values)}/{len(qa_pairs)}")
        else:
            summary[name] = {"count": 0, "mean": None, "min": None, "max": None}
            print(f"  {name}: 无有效评分")

    # --- 6. 保存结果 ---
    output = {
        "summary": summary,
        "details": all_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存至 {output_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="RAGAS 评估脚本 — 对 RAG pipeline 进行多指标自动评估",
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"输入路径：CSV 文件或包含 CSV 文件的目录（默认: {DEFAULT_INPUT}）",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"输出 JSON 文件路径（默认: {DEFAULT_OUTPUT}）",
    )
    return parser.parse_args()


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    args = parse_args()
    run_async(evaluate_all(args.input, args.output))


"""
# 使用默认路径
uv run python -m eval.ragas_eval

# 指定单个 CSV 文件
uv run python -m eval.ragas_eval -i eval/dataset/HAK180烫金机.csv -o eval/hak180_result.json

# 指定整个目录
uv run python -m eval.ragas_eval -i eval/dataset/ -o eval/all_result.json

# 查看帮助
uv run python -m eval.ragas_eval --help
"""

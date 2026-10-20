from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from agents.llm import llm_client
from agents.prompts import load_prompt


async def ainvoke_llm_json(prompt_file_name: str, kwargs: dict):
    prompt = await load_prompt(prompt_file_name)
    prompt_template = PromptTemplate.from_template(prompt)
    chain = prompt_template | llm_client | JsonOutputParser()
    return await chain.ainvoke(kwargs)


async def ainvoke_llm_str(prompt_file_name: str, kwargs: dict):
    prompt = await load_prompt(prompt_file_name)
    prompt_template = PromptTemplate.from_template(prompt)
    chain = prompt_template | llm_client | StrOutputParser()
    return await chain.ainvoke(kwargs)


async def astream_llm_str(prompt_file_name: str, kwargs: dict):
    prompt = await load_prompt(prompt_file_name)
    prompt_template = PromptTemplate.from_template(prompt)
    chain = prompt_template | llm_client
    async for chunk in chain.astream(kwargs):
        """
        chunk.content可能的结构：
        chunk.content = "xxx"
        chunk.content = [
            {"type": "text", "text": "你好"},
            {"type": "text", "text": "，很高兴见到你"}
            "yyy"
        ]
        """
        content = chunk.content
        if content is None:
            continue
        if isinstance(content, str):
            if content:
                yield content
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        yield text
                elif isinstance(item, str):
                    if item:
                        yield item


from pathlib import Path
import aiofiles


async def load_prompt(prompt_file_name: str) -> str:
    file_path = Path(__file__).parent / f"{prompt_file_name}.md"
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        return await f.read()
import asyncio
import shutil
import tempfile
import zipfile
from pathlib import Path
from time import monotonic
from urllib.parse import unquote, urlparse

import aiofiles
import httpx
from pypdf import PdfReader, PdfWriter

from conf import app_config
from core.log import logger


MAX_PDF_PAGES_PER_PART = 180
MINERU_PART_CONCURRENCY = 2
POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 30 * 60
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}


class MineruIntegration:
    def __init__(self):
        self.base_url = app_config.mineru.base_url
        self.token = app_config.mineru.token
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        self.client = httpx.AsyncClient(headers=self.headers, timeout=30)
        self.upload_client = httpx.AsyncClient(timeout=300)

    async def upload_file(self, file_path: Path) -> str:
        url = f"{self.base_url}/file-urls/batch"
        payload = {
            "files": [{"name": file_path.name}],
            "model_version": "vlm",
        }
        signed_url_response = await self.client.post(url, json=payload)
        signed_url_response.raise_for_status()

        resp_data = signed_url_response.json()
        if resp_data.get("code") not in (None, 0):
            raise RuntimeError(
                f"MinerU 创建解析任务失败: {resp_data.get('msg', resp_data)}"
            )
        signed_url = resp_data["data"]["file_urls"][0]
        batch_id = resp_data["data"]["batch_id"]

        async with aiofiles.open(file_path, mode="rb") as fp:
            file_data = await fp.read()

        upload_response = await self.upload_client.put(
            signed_url,
            content=file_data,
        )
        upload_response.raise_for_status()

        poll_url = f"{self.base_url}/extract-results/batch/{batch_id}"
        polled_times = 0
        deadline = monotonic() + POLL_TIMEOUT_SECONDS

        while monotonic() < deadline:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

            poll_response = await self.client.get(poll_url, timeout=30)
            poll_response.raise_for_status()

            poll_data = poll_response.json()
            extract_result = poll_data["data"]["extract_result"]
            if extract_result:
                result_item = extract_result[0]
                state = result_item["state"]
                if state == "done":
                    full_zip_url = result_item.get("full_zip_url")
                    if full_zip_url:
                        return full_zip_url
                    raise RuntimeError(f"{batch_id} 解析完成但未返回结果地址")
                elif state == "failed":
                    error = result_item.get("err_msg", "MinerU 解析失败")
                    raise RuntimeError(f"{batch_id} 解析失败: {error}")
                else:
                    logger.debug(f"{batch_id} 轮询第 {polled_times + 1} 次，状态: {state}")

            polled_times += 1

        raise TimeoutError(
            f"{file_path.name} 的 MinerU 解析等待超过 "
            f"{POLL_TIMEOUT_SECONDS // 60} 分钟，batch_id={batch_id}"
        )

    async def download_markdown(
        self,
        full_zip_url: str,
        output_dir: Path,
        output_stem: str | None = None,
    ) -> Path:
        pdf_stem = output_stem or output_dir.name
        zip_file_name = Path(unquote(urlparse(full_zip_url).path)).name or f"{pdf_stem}_result.zip"

        logger.info(f"开始下载 MinerU 解析结果 ZIP 包: {full_zip_url}")
        response = await self.client.get(full_zip_url, timeout=300)
        response.raise_for_status()

        if output_dir.exists():
            await asyncio.to_thread(shutil.rmtree, output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            prefix=f"{pdf_stem}_",
            suffix=f"_{zip_file_name}",
            delete=False,
        ) as temp_zip_file:
            zip_save_path = Path(temp_zip_file.name)

        try:
            async with aiofiles.open(zip_save_path, mode="wb") as fp:
                await fp.write(response.content)
            logger.info(f"MinerU 解析结果 ZIP 包下载完成: {zip_save_path}")

            def _extract_zip():
                output_dir_resolved = output_dir.resolve()
                with zipfile.ZipFile(zip_save_path, "r") as zip_file:
                    for member in zip_file.infolist():
                        member_path = (output_dir / member.filename).resolve()
                        if member_path != output_dir_resolved and output_dir_resolved not in member_path.parents:
                            raise RuntimeError(f"ZIP 包中包含非法路径: {member.filename}")
                    zip_file.extractall(output_dir)

            await asyncio.to_thread(_extract_zip)
        finally:
            if zip_save_path.exists():
                await asyncio.to_thread(zip_save_path.unlink)

        logger.info(f"MinerU 解析结果 ZIP 包解压完成: {output_dir}")

        md_file_list = list(output_dir.rglob("*.md"))
        if not md_file_list:
            raise FileNotFoundError(f"未找到 Markdown 文件: {output_dir}")

        target_md_file = None
        for md_file in md_file_list:
            if md_file.stem == pdf_stem:
                target_md_file = md_file
                break

        if not target_md_file:
            for md_file in md_file_list:
                if md_file.name.lower() == "full.md":
                    target_md_file = md_file
                    break

        if not target_md_file:
            target_md_file = md_file_list[0]

        if target_md_file.stem != pdf_stem:
            new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
            try:
                target_md_file.rename(new_md_path)
                target_md_file = new_md_path
            except OSError as exc:
                logger.warning(f"Markdown 文件重命名失败，继续使用原文件: {exc}")

        return target_md_file

    async def parse_pdf(self, file_path: Path, output_dir: Path) -> Path:
        """Parse one logical PDF, splitting oversized files before MinerU."""
        page_count = await asyncio.to_thread(self._get_pdf_page_count, file_path)
        if page_count <= MAX_PDF_PAGES_PER_PART:
            full_zip_url = await self.upload_file(file_path)
            return await self.download_markdown(
                full_zip_url,
                output_dir,
                output_stem=file_path.stem,
            )

        if output_dir.exists():
            await asyncio.to_thread(shutil.rmtree, output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        parts_dir = output_dir / "_parts"
        part_paths = await asyncio.to_thread(
            self._split_pdf,
            file_path,
            parts_dir,
        )
        logger.info(
            f"{file_path.name} 共 {page_count} 页，已拆分为 "
            f"{len(part_paths)} 个临时分卷"
        )

        semaphore = asyncio.Semaphore(MINERU_PART_CONCURRENCY)

        async def parse_part(
            part_index: int,
            part_path: Path,
        ) -> tuple[int, Path, Path]:
            async with semaphore:
                logger.info(
                    f"开始解析 {file_path.name} 的分卷 "
                    f"{part_index}/{len(part_paths)}"
                )
                result_dir = parts_dir / f"result_{part_index:04d}"
                full_zip_url = await self.upload_file(part_path)
                markdown_path = await self.download_markdown(
                    full_zip_url,
                    result_dir,
                    output_stem=part_path.stem,
                )
                return part_index, markdown_path, result_dir

        tasks = [
            asyncio.create_task(parse_part(part_index, part_path))
            for part_index, part_path in enumerate(part_paths, start=1)
        ]
        try:
            parsed_parts = await asyncio.gather(*tasks)
            parsed_parts.sort(key=lambda item: item[0])

            return await asyncio.to_thread(
                self._merge_markdown_parts,
                parsed_parts,
                output_dir,
                file_path.stem,
            )
        except Exception:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            if parts_dir.exists():
                await asyncio.to_thread(shutil.rmtree, parts_dir)

    @staticmethod
    def _get_pdf_page_count(file_path: Path) -> int:
        reader = PdfReader(str(file_path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError(f"PDF 已加密，无法解析: {file_path.name}")
        return len(reader.pages)

    @staticmethod
    def _split_pdf(file_path: Path, parts_dir: Path) -> list[Path]:
        parts_dir.mkdir(parents=True, exist_ok=True)
        reader = PdfReader(str(file_path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError(f"PDF 已加密，无法拆分: {file_path.name}")

        part_paths: list[Path] = []
        for part_index, start_page in enumerate(
            range(0, len(reader.pages), MAX_PDF_PAGES_PER_PART),
            start=1,
        ):
            writer = PdfWriter()
            end_page = min(start_page + MAX_PDF_PAGES_PER_PART, len(reader.pages))
            for page_index in range(start_page, end_page):
                writer.add_page(reader.pages[page_index])

            part_path = parts_dir / f"{file_path.stem}.part_{part_index:04d}.pdf"
            with part_path.open("wb") as output_file:
                writer.write(output_file)
            part_paths.append(part_path)

        return part_paths

    @staticmethod
    def _merge_markdown_parts(
        parsed_parts: list[tuple[int, Path, Path]],
        output_dir: Path,
        output_stem: str,
    ) -> Path:
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        merged_contents: list[str] = []

        for part_index, markdown_path, result_dir in parsed_parts:
            content = markdown_path.read_text(encoding="utf-8")
            image_paths = sorted(
                path
                for path in result_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            for image_path in image_paths:
                merged_name = f"part_{part_index:04d}_{image_path.name}"
                shutil.copy2(image_path, images_dir / merged_name)
                content = content.replace(image_path.name, merged_name)

            merged_contents.append(content.strip())

        merged_path = output_dir / f"{output_stem}.md"
        merged_path.write_text(
            "\n\n".join(merged_contents) + "\n",
            encoding="utf-8",
        )
        return merged_path


mineru_integration = MineruIntegration()


if __name__ == "__main__":
    async def test():
        full_zip_url = await mineru_integration.upload_file(Path("uploads/test.pdf"))
        logger.info(full_zip_url)
        md_file_path = await mineru_integration.download_markdown(full_zip_url, Path("uploads/"))  # type: ignore[arg-type]
        logger.info(md_file_path)

    asyncio.run(test())

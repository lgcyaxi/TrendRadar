# coding=utf-8
"""
Ollama Client for WeWork Compact Extension

Simple Ollama client for generating AI-powered news summaries.
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class OllamaClient:
    """Simple Ollama client for generating news summaries."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        prompt_file: Optional[str] = None,
    ) -> None:
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server base URL
            model: Model name to use
            timeout: Request timeout in seconds
            prompt_file: Optional custom prompt file path
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._prompt_template = ""
        self._load_prompt(prompt_file)

    def _load_prompt(self, prompt_file: Optional[str]) -> None:
        """Load prompt template from file or use default."""
        if not prompt_file:
            self._prompt_template = self._default_prompt()
            return

        # Support CONFIG_DIR env var
        config_dir_env = os.environ.get("CONFIG_DIR")
        if config_dir_env:
            config_dir = Path(config_dir_env)
        else:
            config_dir = Path(__file__).parent.parent.parent / "config"

        prompt_path = config_dir / prompt_file

        if prompt_path.exists():
            try:
                self._prompt_template = prompt_path.read_text(encoding="utf-8")
                logger.debug("[wework_compact] Loaded prompt from: {}", prompt_path)
            except Exception as e:
                logger.warning("[wework_compact] Failed to load prompt {}: {}", prompt_path, e)
                self._prompt_template = self._default_prompt()
        else:
            logger.warning("[wework_compact] Prompt file not found: {}", prompt_path)
            self._prompt_template = self._default_prompt()

    def _default_prompt(self) -> str:
        """Return default prompt template."""
        return """你是一位新闻编辑。请根据以下新闻标题列表，撰写一段简短的新闻简报（约250字，2-3句话），概括今日发生的主要事件。

要求：
1. 采用新闻播报风格：客观、事实性、简洁
2. 概括今日发生的核心事件和事实
3. 使用新闻开篇的叙述方式（如"今日，..."、"据报道，..."）
4. 不要逐条列举标题，而是提炼出整体事件画面
5. 避免主观分析和评论，只陈述事实
6. 如果有多个独立热点，用分号分隔
7. 语言简练，避免冗余

新闻标题：
{news_titles}

请直接输出新闻简报内容，不需要任何前缀或格式标记："""

    def _build_prompt(self, news_titles: List[str]) -> str:
        """Build complete prompt with news titles."""
        titles_text = "\n".join(f"- {title}" for title in news_titles[:30])
        return self._prompt_template.replace("{news_titles}", titles_text)

    def _request(
        self, method: str, path: str, payload: Optional[Dict[str, Any]]
    ) -> Any:
        """Make HTTP request to Ollama."""
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        data = None

        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8")

        if not body:
            return {}

        return json.loads(body)

    def generate_summary(self, news_titles: List[str]) -> Optional[str]:
        """
        Generate a summary for the given news titles.

        Args:
            news_titles: List of news titles to summarize

        Returns:
            Generated summary text, or None if failed
        """
        if not news_titles:
            return None

        prompt = self._build_prompt(news_titles)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "thinking": False,  # Disable thinking mode for GLM models
        }

        try:
            logger.info("[wework_compact] Calling Ollama for summary ({} titles)", len(news_titles))

            data = self._request("POST", "/api/generate", payload)

            if isinstance(data, dict):
                response_text = data.get("response", "")
                if response_text:
                    # Try to parse JSON first
                    try:
                        result = json.loads(response_text)
                        if isinstance(result, dict) and "summary" in result:
                            return result["summary"].strip()
                        if isinstance(result, dict) and "content" in result:
                            return result["content"].strip()
                    except json.JSONDecodeError:
                        pass

                    # Fallback to raw text
                    return response_text.strip()

            logger.warning("[wework_compact] Empty response from Ollama")
            return None

        except urllib.error.HTTPError as e:
            logger.error(
                "[wework_compact] Ollama HTTP error: {} - {}",
                e.code,
                e.reason,
            )
            return None
        except urllib.error.URLError as e:
            logger.error("[wework_compact] Ollama connection error: {}", e.reason)
            return None
        except Exception as e:
            logger.error(
                "[wework_compact] Ollama request failed: {} - {}", type(e).__name__, e
            )
            return None

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            self._request("GET", "/api/tags", None)
            return True
        except Exception as e:
            logger.debug("[wework_compact] Ollama not available: {}", e)
            return False

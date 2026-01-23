# coding=utf-8
"""
NewTitlesHandler for WeWork Compact Extension

Handles the "本日新增" (New Titles) section with AI summary generation.
Uses Ollama client for AI-powered cross-news summaries.
"""

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..ollama_client import OllamaClient
from .base import BaseHandler, HandlerResult


class NewTitlesHandler(BaseHandler):
    """
    Handler for the new titles section (本日新增).

    Features:
    - Extracts newly detected titles from report_data
    - Generates AI-powered cross-news summary using Ollama (dynamic)
    - Uses dedicated ollama config (like report_dedupe extension)
    """

    # Default prompt template (fallback if prompt file not loaded)
    _DEFAULT_PROMPT = """你是一位新闻编辑。请根据以下新闻标题列表，撰写一段简短的新闻简报（约250字，2-3句话），概括今日发生的主要事件。

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

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize handler with configuration."""
        super().__init__(config)
        self._ollama_client: Optional[OllamaClient] = None
        self._prompt_template: str = self._DEFAULT_PROMPT

    def _build_prompt(self, titles_text: str) -> str:
        """Build complete prompt with news titles text."""
        return self._prompt_template.replace("{news_titles}", titles_text)

    @property
    def section_name(self) -> str:
        return "new_titles"

    @property
    def section_title(self) -> str:
        return "今日热点"

    def process(
        self,
        context: Any,
        main_config: Dict[str, Any],
    ) -> HandlerResult:
        """
        Process the new titles section.

        Args:
            context: Application context with report_data, ai_analysis
            main_config: Main application config (for AI settings)

        Returns:
            HandlerResult with formatted content
        """
        try:
            # Get report_data from context
            report_data = getattr(context, 'report_data', None)
            if report_data is None:
                if hasattr(context, 'get'):
                    report_data = context.get('report_data', {})
                else:
                    return HandlerResult(
                        section_name=self.section_name,
                        title=self.section_title,
                        content="",
                        success=False,
                        error="No report_data found in context",
                    )

            if not report_data:
                return HandlerResult(
                    section_name=self.section_name,
                    title=self.section_title,
                    content="",
                    success=False,
                    error="Empty report_data",
                )

            # Extract NEW titles with platforms
            titles_with_platforms = self._extract_new_titles(report_data)

            if not titles_with_platforms:
                # Return explicit message when no new hot topics
                empty_message = self._config.get('empty_message', '暂无新增今日热点')
                return HandlerResult(
                    section_name=self.section_name,
                    title=self.section_title,
                    content=empty_message,
                    item_count=0,
                    success=True,
                )

            # Get Ollama client if AI summary is enabled
            ollama_client = None
            ai_summary_enabled = self._config.get('include_ai_summary', True)
            logger.info("[new_titles] AI summary enabled: {}, trying to get Ollama client...", ai_summary_enabled)

            if ai_summary_enabled:
                ollama_client = self._get_ai_client(main_config)
                if ollama_client:
                    logger.info("[new_titles] Ollama client ready, generating AI summary for {} titles...", len(titles_with_platforms))
                else:
                    logger.warning("[new_titles] Ollama client not available, will use simple format")

            # Generate content
            if ollama_client:
                content = self._generate_ai_mode_content(
                    titles_with_platforms,
                    ollama_client,
                )
                if content:
                    logger.info("[new_titles] AI summary generated successfully ({} chars)", len(content))
                    return HandlerResult(
                        section_name=self.section_name,
                        title=self.section_title,
                        content=content,
                        item_count=len(titles_with_platforms),
                        success=True,
                    )
                logger.warning("[new_titles] AI mode failed (empty response), falling back to simple mode")

            # Fallback mode - simple format without global AI analysis
            # (Global AI analysis is for current_list section, not for new_titles)
            logger.info("[new_titles] Using simple format (no AI summary) for {} titles", len(titles_with_platforms))
            content = self._simple_format(titles_with_platforms)

            return HandlerResult(
                section_name=self.section_name,
                title=self.section_title,
                content=content,
                item_count=len(titles_with_platforms),
                success=True,
            )

        except Exception as e:
            logger.error("[new_titles] Error processing: {}", e)
            return HandlerResult(
                section_name=self.section_name,
                title=self.section_title,
                content="",
                success=False,
                error=str(e),
            )

    def _extract_new_titles(
        self, report_data: Dict[str, Any]
    ) -> List[Tuple[str, List[str]]]:
        """
        Extract newly detected titles with their platforms.

        Extracts directly from report_data['new_titles'] which contains
        titles that were newly detected and matched keyword filters.

        Supports two formats:
        1. List format: [{source_name: "微博", titles: [{title, ...}]}, ...]
        2. Dict format: {source_id: {title: title_data, ...}, ...}

        Args:
            report_data: Report data dictionary

        Returns:
            List of (title, [platforms]) tuples
        """
        max_titles = self._config.get('max_titles', 20)
        max_platform_display = self._config.get('max_platform_display', 3)

        # Build a map of title -> platforms by aggregating across sources
        # This handles titles that appear on multiple platforms
        title_platforms: Dict[str, List[str]] = {}

        new_titles_data = report_data.get('new_titles', [])

        # Handle list format: [{source_name: "微博", titles: [...]}, ...]
        if isinstance(new_titles_data, list):
            for source_data in new_titles_data:
                if not isinstance(source_data, dict):
                    continue

                source_name = source_data.get('source_name', '')
                titles_list = source_data.get('titles', [])

                for title_info in titles_list:
                    if not isinstance(title_info, dict):
                        continue

                    title = title_info.get('title', '')
                    if not title:
                        continue

                    title_clean = self._clean_title(title)
                    if not title_clean:
                        continue

                    # Get platform from title_info or fallback to source
                    platform = title_info.get('source_name', '') or source_name

                    # Aggregate platforms for the same title
                    if title_clean not in title_platforms:
                        title_platforms[title_clean] = []

                    if platform and platform not in title_platforms[title_clean]:
                        title_platforms[title_clean].append(platform)

        # Handle dict format: {source_id: {title: title_data}, ...}
        elif isinstance(new_titles_data, dict):
            for source_id, titles_dict in new_titles_data.items():
                if not isinstance(titles_dict, dict):
                    continue

                source_name = source_id
                # Try to get a better name from id_to_name if available
                # (but we don't have access to it here, so just use source_id)

                for title, title_info in titles_dict.items():
                    if not isinstance(title_info, dict):
                        continue

                    title_clean = self._clean_title(title)
                    if not title_clean:
                        continue

                    # Get platform from title_info
                    platform = title_info.get('source_name', '') or source_name

                    # Aggregate platforms for the same title
                    if title_clean not in title_platforms:
                        title_platforms[title_clean] = []

                    if platform and platform not in title_platforms[title_clean]:
                        title_platforms[title_clean].append(platform)

        # Convert to result list
        results: List[Tuple[str, List[str]]] = []
        for title_clean, platforms in title_platforms.items():
            if len(results) >= max_titles:
                break

            # Normalize and limit platforms
            normalized_platforms = self._normalize_platform_names(platforms)
            results.append((title_clean, normalized_platforms[:max_platform_display]))

        return results

    def _get_ai_client(self, main_config: Dict[str, Any] = None) -> Optional[Any]:
        """
        Get AI client for the extension.

        Args:
            main_config: Main application config (contains context)

        Returns:
            AI client instance if config is valid, None otherwise
        """
        if self._ollama_client is not None:
            return self._ollama_client

        # Use extension's ollama config
        ollama_config = self._config.get('ollama', {})
        model = ollama_config.get('model', '').strip()

        if not model:
            logger.debug("[new_titles] AI: No model configured")
            return None

        # 优先使用主 AI 客户端
        if main_config:
            try:
                from extensions.ai_client import create_ai_client
                self._ollama_client = create_ai_client(
                    main_config,
                    model=model,
                    thinking=False  # 禁用 thinking 模式
                )
                if self._ollama_client:
                    logger.info("[new_titles] Using main AI client with Ollama model: {}", model)
                    return self._ollama_client
            except Exception as e:
                logger.debug("[new_titles] Could not create AI client: {}", e)

        logger.debug("[new_titles] AI client not available, AI summary disabled")
        return None

    def _generate_cross_news_summary(
        self, titles: List[str], ai_client: Any
    ) -> Optional[str]:
        """
        Generate a holistic summary across all news using AI client.

        Args:
            titles: List of news titles to summarize
            ai_client: AI client instance (AIClient or OllamaClient)

        Returns:
            Generated summary text, or None if failed
        """
        if not titles:
            return None

        # 获取模型覆盖（用于降级时的 OllamaClient）
        model = getattr(self, '_ollama_model_override', None)

        # 检查是否是 AIClient（LiteLLM）
        from trendradar.ai.client import AIClient as LiteAIClient
        if isinstance(ai_client, LiteAIClient):
            # 使用 LiteLLM AIClient
            titles_text = "\n".join(f"- {title}" for title in titles[:30])
            prompt = self._build_prompt(titles_text)

            try:
                messages = [{"role": "user", "content": prompt}]
                response = ai_client.chat(messages)
                if response:
                    logger.debug("[new_titles] AI: Summary generated successfully ({} chars)", len(response))
                    return response.strip()
            except Exception as e:
                logger.warning("[new_titles] AI: Summary generation failed: {}", e)
                return None
        else:
            # 使用 OllamaClient（降级）
            summary = ai_client.generate_summary(titles, model=model)
            if summary:
                logger.debug("[new_titles] AI: Summary generated successfully ({} chars)", len(summary))
            else:
                logger.warning("[new_titles] AI: Summary generation failed")
            return summary

    def _generate_ai_mode_content(
        self,
        titles_with_platforms: List[Tuple[str, List[str]]],
        ollama_client: OllamaClient,
    ) -> Optional[str]:
        """
        Generate content in AI mode with cross-news summary.

        Args:
            titles_with_platforms: List of (title, platforms) tuples
            ollama_client: OllamaClient for generating summaries

        Returns:
            Formatted content string, or None if AI generation failed
        """
        titles = [t[0] for t in titles_with_platforms]

        summary = self._generate_cross_news_summary(titles, ollama_client)
        if not summary:
            return None

        lines = []
        count = len(titles_with_platforms)
        lines.append(f"{self.section_title}: 共发现 {count} 条热点新闻")
        lines.append("")
        lines.append(summary)
        lines.append("")

        max_titles = self._config.get('max_titles', 20)
        max_platform_display = self._config.get('max_platform_display', 3)

        for title, platforms in titles_with_platforms[:max_titles]:
            lines.append(self._format_title_line(title, platforms, max_platform_display))

        return '\n'.join(lines)

    def _simple_format(
        self,
        titles_with_platforms: List[Tuple[str, List[str]]],
    ) -> str:
        """Generate simple format with titles only.

        Used as fallback when AI summary generation fails or is disabled.
        Only shows title count and list, no AI analysis (avoid duplication).
        """
        lines = []

        count = len(titles_with_platforms)
        lines.append(f"{self.section_title}: 共发现 {count} 条热点新闻")
        lines.append("")

        max_titles = self._config.get('max_titles', 20)
        max_platform_display = self._config.get('max_platform_display', 3)

        for title, platforms in titles_with_platforms[:max_titles]:
            lines.append(self._format_title_line(title, platforms, max_platform_display))

        return '\n'.join(lines)

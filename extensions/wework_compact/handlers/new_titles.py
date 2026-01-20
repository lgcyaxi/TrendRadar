# coding=utf-8
"""
NewTitlesHandler for WeWork Compact Extension

Handles the "本日新增" (New Titles) section with AI summary generation.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from loguru import logger

from .base import BaseHandler, HandlerResult


class NewTitlesHandler(BaseHandler):
    """
    Handler for the new titles section (本日新增).

    Features:
    - Extracts newly detected titles from report_data
    - Generates AI-powered cross-news summary (dynamic)
    - Appends static AI analysis section if available
    """

    @property
    def section_name(self) -> str:
        return "new_titles"

    @property
    def section_title(self) -> str:
        return "本日新增"

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
                return HandlerResult(
                    section_name=self.section_name,
                    title=self.section_title,
                    content="",
                    item_count=0,
                    success=True,
                )

            # Get AI config
            ai_config = self._get_ai_config(main_config)

            # Generate content
            if ai_config and self._config.get('include_ai_summary', True):
                content = self._generate_ai_mode_content(
                    titles_with_platforms,
                    ai_config,
                )
                if content:
                    return HandlerResult(
                        section_name=self.section_name,
                        title=self.section_title,
                        content=content,
                        item_count=len(titles_with_platforms),
                        success=True,
                    )
                logger.warning("[new_titles] AI mode failed, falling back to simple mode")

            # Fallback mode
            content = self._fallback_format(titles_with_platforms)

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
        seen_cleaned: Dict[str, str] = {}  # cleaned_title -> original_title

        new_titles_data = report_data.get('new_titles', [])
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
                    seen_cleaned[title_clean] = title_clean

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

    def _get_ai_config(self, main_config: Dict) -> Optional[Dict]:
        """Get AI config from main config."""
        ai_settings = self._config.get('ai', {})
        if not ai_settings.get('use_main_config', True):
            return None

        ai_analysis_config = main_config.get('AI_ANALYSIS', {})
        if not ai_analysis_config.get('ENABLED', False):
            return None

        ai_model_config = main_config.get('AI', {})

        api_key = ai_model_config.get('API_KEY') or os.environ.get('AI_API_KEY', '')
        if not api_key:
            return None

        return {**ai_analysis_config, **ai_model_config}

    def _load_prompt_template(self) -> str:
        """Load the custom prompt template for compact summary generation."""
        ai_settings = self._config.get('ai', {})
        prompt_file = ai_settings.get('prompt_file', 'wework_compact_prompt.txt')
        config_dir = Path(__file__).parent.parent.parent.parent / "config"
        prompt_path = config_dir / prompt_file

        if prompt_path.exists():
            return prompt_path.read_text(encoding='utf-8')

        return self._build_default_prompt()

    def _build_default_prompt(self) -> str:
        """Build default prompt dynamically from AI summary config."""
        ai_settings = self._config.get('ai', {})
        ai_summary_config = ai_settings.get('ai_summary', {})

        tone_map = {
            "journalistic": "新闻播报风格：客观、事实性、简洁",
            "analytical": "分析评论风格：深入解读、有洞察力",
            "conversational": "口语化风格：轻松、易懂、亲和",
        }

        focus_map = {
            "facts": "概括今日发生的核心事件和事实",
            "analysis": "分析事件背后的原因和影响",
            "trends": "提炼整体趋势和关联",
            "all": "综合事件、分析和趋势",
        }

        target_length = ai_summary_config.get("target_length", 250)
        tone_instruction = tone_map.get(
            ai_summary_config.get("tone", "journalistic"),
            tone_map["journalistic"]
        )
        focus_instruction = focus_map.get(
            ai_summary_config.get("focus", "facts"),
            focus_map["facts"]
        )

        prompt = f"""你是一位新闻编辑。请根据以下新闻标题列表，撰写一段简短的新闻简报（约{target_length}字，2-3句话），概括今日发生的主要事件。

要求：
1. 采用{tone_instruction}
2. {focus_instruction}
3. 使用新闻开篇的叙述方式（如"今日，..."、"据报道，..."）
4. 不要逐条列举标题，而是提炼出整体事件画面
5. 避免主观分析和评论，只陈述事实
6. 如果有多个独立热点，用分号分隔
7. 语言简练，避免冗余

新闻标题：
{{news_titles}}

请直接输出新闻简报内容，不需要任何前缀或格式标记："""

        return prompt

    def _generate_cross_news_summary(
        self, titles: List[str], ai_config: Dict
    ) -> Optional[str]:
        """Generate a holistic summary across all news using AI."""
        try:
            prompt_template = self._load_prompt_template()
            news_titles = '\n'.join(f"- {title}" for title in titles[:30])
            prompt = prompt_template.replace('{news_titles}', news_titles)

            api_key = ai_config.get('API_KEY') or os.environ.get('AI_API_KEY', '')
            provider = ai_config.get('PROVIDER', 'openai')
            model = ai_config.get('MODEL', 'gpt-4o-mini')
            base_url = ai_config.get('BASE_URL', '')
            timeout = ai_config.get('TIMEOUT', 30)

            if base_url:
                if base_url.rstrip('/').endswith('/chat/completions'):
                    api_url = base_url.rstrip('/')
                else:
                    api_url = f"{base_url.rstrip('/')}/chat/completions"
            elif provider == 'deepseek':
                api_url = "https://api.deepseek.com/chat/completions"
            else:
                api_url = "https://api.openai.com/v1/chat/completions"

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}',
            }

            max_tokens = ai_config.get('MAX_TOKENS', 500)

            payload = {
                'model': model,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': max_tokens,
                'temperature': 0.7,
            }

            logger.debug("[new_titles] AI request: url={}, model={}", api_url, model)

            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            if response.status_code != 200:
                logger.warning("[new_titles] AI response status: {}, body: {}",
                             response.status_code, response.text[:500])

            response.raise_for_status()

            result = response.json()
            summary = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            return summary.strip() if summary else None

        except requests.exceptions.RequestException as e:
            logger.error("[new_titles] AI request failed: {} - {}", type(e).__name__, e)
            return None
        except Exception as e:
            logger.error("[new_titles] AI summary generation failed: {} - {}", type(e).__name__, e)
            return None

    def _generate_ai_mode_content(
        self,
        titles_with_platforms: List[Tuple[str, List[str]]],
        ai_config: Dict,
    ) -> Optional[str]:
        """Generate content in AI mode with cross-news summary."""
        titles = [t[0] for t in titles_with_platforms]

        summary = self._generate_cross_news_summary(titles, ai_config)
        if not summary:
            return None

        lines = []
        count = len(titles_with_platforms)
        lines.append(f"{self.section_title}: 共发现 {count} 条新热点")
        lines.append("")
        lines.append(summary)
        lines.append("")

        max_titles = self._config.get('max_titles', 20)
        max_platform_display = self._config.get('max_platform_display', 3)

        for title, platforms in titles_with_platforms[:max_titles]:
            lines.append(self._format_title_line(title, platforms, max_platform_display))

        return '\n'.join(lines)

    def _fallback_format(
        self,
        titles_with_platforms: List[Tuple[str, List[str]]],
    ) -> str:
        """Generate compact format without AI (fallback mode)."""
        lines = []

        count = len(titles_with_platforms)
        summary = f"共发现 {count} 条新热点" if count > 0 else "暂无新热点"
        lines.append(f"{self.section_title}: {summary}")
        lines.append("")

        max_titles = self._config.get('max_titles', 20)
        max_platform_display = self._config.get('max_platform_display', 3)

        for title, platforms in titles_with_platforms[:max_titles]:
            lines.append(self._format_title_line(title, platforms, max_platform_display))

        return '\n'.join(lines)

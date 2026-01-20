# coding=utf-8
"""
CurrentListHandler for WeWork Compact Extension

Handles the "当前榜单" (Current List) section - all trending titles currently on lists.
"""

from typing import Any, Dict, List, Optional, Tuple

from .base import BaseHandler, HandlerResult


class CurrentListHandler(BaseHandler):
    """
    Handler for the current trending list section (当前榜单).

    Extracts all titles from report_data['stats'] and formats them without URLs.
    """

    @property
    def section_name(self) -> str:
        return "current_list"

    @property
    def section_title(self) -> str:
        return "当前榜单"

    def process(
        self,
        context: Any,
        main_config: Dict[str, Any],
    ) -> HandlerResult:
        """
        Process the current list section.

        Args:
            context: Application context with report_data
            main_config: Main application config

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

            # Extract titles with platforms
            titles_with_platforms = self._extract_current_titles(report_data)

            if not titles_with_platforms:
                return HandlerResult(
                    section_name=self.section_name,
                    title=self.section_title,
                    content="",
                    item_count=0,
                    success=True,
                )

            # Get existing AI analysis
            existing_ai_analysis = self._get_existing_ai_analysis(context)

            # Format content
            content = self._format_content(titles_with_platforms, existing_ai_analysis)

            return HandlerResult(
                section_name=self.section_name,
                title=self.section_title,
                content=content,
                item_count=len(titles_with_platforms),
                success=True,
            )

        except Exception as e:
            return HandlerResult(
                section_name=self.section_name,
                title=self.section_title,
                content="",
                success=False,
                error=str(e),
            )

    def _extract_current_titles(
        self, report_data: Dict[str, Any]
    ) -> List[Tuple[str, List[str]]]:
        """
        Extract all current titles with their platforms from stats.

        Args:
            report_data: Report data dictionary

        Returns:
            List of (title, [platforms]) tuples
        """
        results: List[Tuple[str, List[str]]] = []
        seen_titles: set = set()

        # Get max titles from config
        max_titles = self._config.get('max_titles', 50)
        max_platform_display = self._config.get('max_platform_display', 3)

        stats = report_data.get('stats', [])
        for stat in stats:
            titles = stat.get('titles', [])
            for title_info in titles:
                if not isinstance(title_info, dict):
                    continue

                title = title_info.get('title', '')
                if not title:
                    continue

                # Clean title
                title_clean = self._clean_title(title)
                if not title_clean or title_clean in seen_titles:
                    continue

                seen_titles.add(title_clean)

                # Get platforms
                platforms = title_info.get('platforms', [])
                if not platforms:
                    platform = title_info.get('platform', '')
                    source_name = title_info.get('source_name', '')
                    if platform:
                        platforms = [platform]
                    elif source_name:
                        platforms = [source_name]

                platforms = self._normalize_platform_names(platforms)
                results.append((title_clean, platforms[:max_platform_display]))

                if len(results) >= max_titles:
                    break

            if len(results) >= max_titles:
                break

        return results

    def _get_existing_ai_analysis(self, context: Any) -> Optional[str]:
        """Get the existing AI analysis content from context if available."""
        ai_content = getattr(context, 'ai_content', None)
        if ai_content:
            return ai_content

        ai_analysis = getattr(context, 'ai_analysis', None)
        if ai_analysis:
            if getattr(ai_analysis, 'success', False):
                return self._format_ai_analysis_section(ai_analysis)

        return None

    def _format_ai_analysis_section(self, ai_analysis: Any) -> str:
        """Format AI analysis result into a readable section."""
        lines = []
        lines.append("\n━━━━━━━━━━━━━━")
        lines.append("📊 AI 深度分析")
        lines.append("━━━━━━━━━━━━━━")

        if getattr(ai_analysis, 'core_trends', ''):
            lines.append(f"\n📌 核心热点\n{ai_analysis.core_trends}")

        if getattr(ai_analysis, 'sentiment_controversy', ''):
            lines.append(f"\n💭 舆论风向\n{ai_analysis.sentiment_controversy}")

        if getattr(ai_analysis, 'signals', ''):
            lines.append(f"\n⚡ 异动信号\n{ai_analysis.signals}")

        if getattr(ai_analysis, 'rss_insights', ''):
            lines.append(f"\n📰 RSS 洞察\n{ai_analysis.rss_insights}")

        if getattr(ai_analysis, 'outlook_strategy', ''):
            lines.append(f"\n📝 策略建议\n{ai_analysis.outlook_strategy}")

        return '\n'.join(lines)

    def _format_content(
        self,
        titles_with_platforms: List[Tuple[str, List[str]]],
        existing_ai_analysis: Optional[str] = None,
    ) -> str:
        """
        Format the current list content.

        Args:
            titles_with_platforms: List of (title, platforms) tuples
            existing_ai_analysis: Optional AI analysis to append

        Returns:
            Formatted content string
        """
        lines = []
        count = len(titles_with_platforms)

        # Header
        lines.append(f"{self.section_title} (共{count}条)")
        lines.append("")

        # Title lines
        max_platform_display = self._config.get('max_platform_display', 3)
        for title, platforms in titles_with_platforms:
            lines.append(self._format_title_line(title, platforms, max_platform_display))

        # Append AI analysis if available and configured
        include_ai_analysis = self._config.get('include_ai_analysis', True)
        if include_ai_analysis and existing_ai_analysis:
            lines.append("")
            lines.append(existing_ai_analysis)

        return '\n'.join(lines)

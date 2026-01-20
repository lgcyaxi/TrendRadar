# coding=utf-8
"""
StandaloneHandler for WeWork Compact Extension

Handles the "独立展示区" (Standalone Display) section.
"""

from typing import Any, Dict, List

from .base import BaseHandler, HandlerResult


class StandaloneHandler(BaseHandler):
    """
    Handler for the standalone display section (独立展示区).

    Extracts from standalone_data which contains:
    - platforms: Specific hot list platforms (e.g., weibo, zhihu)
    - rss_feeds: Specific RSS feeds (e.g., 36kr, ithome)
    """

    @property
    def section_name(self) -> str:
        return "standalone"

    @property
    def section_title(self) -> str:
        return "独立展示区"

    def process(
        self,
        context: Any,
        main_config: Dict[str, Any],
    ) -> HandlerResult:
        """
        Process the standalone display section.

        Args:
            context: Application context with standalone_data
            main_config: Main application config

        Returns:
            HandlerResult with formatted content
        """
        try:
            # Get standalone_data from context
            standalone_data = getattr(context, 'standalone_data', None)
            if standalone_data is None:
                if hasattr(context, 'get'):
                    standalone_data = context.get('standalone_data', None)

            if not standalone_data:
                return HandlerResult(
                    section_name=self.section_name,
                    title=self.section_title,
                    content="",
                    item_count=0,
                    success=True,
                )

            # Format content
            content, item_count = self._format_content(standalone_data)

            if not content:
                return HandlerResult(
                    section_name=self.section_name,
                    title=self.section_title,
                    content="",
                    item_count=0,
                    success=True,
                )

            return HandlerResult(
                section_name=self.section_name,
                title=self.section_title,
                content=content,
                item_count=item_count,
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

    def _format_content(self, standalone_data: Dict[str, Any]) -> tuple:
        """
        Format the standalone display content.

        Args:
            standalone_data: Standalone data dictionary with platforms and rss_feeds

        Returns:
            Tuple of (formatted content string, total item count)
        """
        lines = []
        total_count = 0

        platforms = standalone_data.get("platforms", [])
        rss_feeds = standalone_data.get("rss_feeds", [])

        if not platforms and not rss_feeds:
            return "", 0

        # Header
        lines.append(self.section_title)
        lines.append("")

        # Process platforms (hot lists)
        for platform in platforms:
            platform_name = platform.get("name", platform.get("id", "Unknown"))
            items = platform.get("items", [])

            if not items:
                continue

            # Platform header with emoji
            emoji = self._get_platform_emoji(platform.get("id", ""))
            lines.append(f"{emoji} {platform_name}")

            # Format items
            max_items = self._config.get('max_items_per_source', 10)
            for item in items[:max_items]:
                title = item.get("title", "")
                if not title:
                    continue

                title_clean = self._clean_title(title)
                if not title_clean:
                    continue

                # Format ranks if available
                ranks = item.get("ranks", [])
                rank_str = self._format_rank_range(ranks)

                if rank_str:
                    lines.append(f"- {title_clean} ({rank_str})")
                else:
                    lines.append(f"- {title_clean}")

                total_count += 1

            lines.append("")  # Empty line between sources

        # Process RSS feeds
        for feed in rss_feeds:
            feed_name = feed.get("name", feed.get("id", "Unknown"))
            items = feed.get("items", [])

            if not items:
                continue

            # Feed header with emoji
            lines.append(f"📰 {feed_name}")

            # Format items
            max_items = self._config.get('max_items_per_source', 10)
            for item in items[:max_items]:
                title = item.get("title", "")
                if not title:
                    continue

                title_clean = self._clean_title(title)
                if not title_clean:
                    continue

                lines.append(f"- {title_clean}")
                total_count += 1

            lines.append("")  # Empty line between sources

        # Remove trailing empty lines
        while lines and lines[-1] == "":
            lines.pop()

        return '\n'.join(lines), total_count

    def _get_platform_emoji(self, platform_id: str) -> str:
        """Get emoji for a platform."""
        emoji_map = {
            'weibo': '📊',
            'zhihu': '💬',
            'baidu': '🔍',
            'douyin': '🎵',
            'bilibili': '📺',
            'toutiao': '📱',
            'weixin': '💚',
            'kuaishou': '🎬',
            'xiaohongshu': '📕',
            'thepaper': '📰',
        }
        return emoji_map.get(platform_id.lower(), '📊')

    def _format_rank_range(self, ranks: List[int]) -> str:
        """
        Format rank range for display.

        Examples:
        - [1] -> "排名1"
        - [1, 2, 3] -> "排名1-3"
        - [5, 3, 1] -> "排名1-5"
        """
        if not ranks:
            return ""

        valid_ranks = [r for r in ranks if isinstance(r, int) and r > 0]
        if not valid_ranks:
            return ""

        min_rank = min(valid_ranks)
        max_rank = max(valid_ranks)

        if min_rank == max_rank:
            return f"排名{min_rank}"
        return f"排名{min_rank}-{max_rank}"

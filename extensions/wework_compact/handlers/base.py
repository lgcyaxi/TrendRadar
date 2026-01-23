# coding=utf-8
"""
Base Handler for WeWork Compact Extension

Provides the abstract base class and result dataclass for all section handlers.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class HandlerResult:
    """Result from a section handler."""

    section_name: str
    title: str
    content: str
    item_count: int = 0
    success: bool = True
    error: Optional[str] = None

    def __bool__(self) -> bool:
        """Return True if handler produced valid content."""
        return self.success and bool(self.content)


class BaseHandler(ABC):
    """
    Abstract base class for section handlers.

    Each handler is responsible for processing a specific section of the notification:
    - current_list: Current trending titles (当前榜单)
    - new_titles: Newly detected titles (本日新增)
    - standalone: Standalone display section (独立展示区)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize handler with configuration.

        Args:
            config: Handler-specific configuration from wework_compact.yaml
        """
        self._config = config

    @property
    @abstractmethod
    def section_name(self) -> str:
        """Unique identifier for this section (e.g., 'current_list')."""
        pass

    @property
    @abstractmethod
    def section_title(self) -> str:
        """Display title for this section (e.g., '当前榜单')."""
        pass

    @abstractmethod
    def process(
        self,
        context: Any,
        main_config: Dict[str, Any],
    ) -> HandlerResult:
        """
        Process the section and return formatted content.

        Args:
            context: Application context with report_data, ai_analysis, etc.
            main_config: Main application config (for AI settings, etc.)

        Returns:
            HandlerResult with formatted content
        """
        pass

    # === Shared Utility Methods ===

    def _strip_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text).strip()

    def _strip_markdown_links(self, text: str) -> str:
        """Convert [title](url) to just title."""
        link_pattern = r'\[([^\]]+)\]\([^\)]+\)'
        return re.sub(link_pattern, r'\1', text)

    def _clean_title(self, title: str) -> str:
        """Clean a title by removing URLs and markdown links."""
        title = self._strip_markdown_links(title)
        title = self._strip_urls(title)
        return title.strip()

    def _normalize_platform_names(self, platforms: List[str]) -> List[str]:
        """Normalize platform names to readable short names."""
        name_map = {
            'weibo': '微博',
            'zhihu': '知乎',
            'baidu': '百度',
            'douyin': '抖音',
            'bilibili': 'B站',
            'toutiao': '头条',
            '36kr': '36氪',
            'ithome': 'IT之家',
            'v2ex': 'V2EX',
            'juejin': '掘金',
            'huxiu': '虎嗅',
            'weixin': '微信',
            'kuaishou': '快手',
            'xiaohongshu': '小红书',
            'thepaper': '澎湃',
        }

        normalized = []
        for p in platforms:
            if not p:
                continue
            p_lower = p.lower().strip()
            if p_lower in name_map:
                normalized.append(name_map[p_lower])
            else:
                normalized.append(p.strip())

        return normalized

    def _format_title_line(
        self,
        title: str,
        platforms: List[str],
        max_platform_display: int = 3,
    ) -> str:
        """Format a title line with platforms."""
        if platforms:
            platform_str = '/'.join(platforms[:max_platform_display])
            return f"- {title} ({platform_str})"
        return f"- {title}"

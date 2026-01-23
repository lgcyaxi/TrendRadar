# coding=utf-8
"""
Report Dedupe Plugin

Plugin that merges similar news titles from different platforms,
reducing duplicates and presenting cleaner results.

Also implements HTMLRenderHook to:
- Add multi-platform HTML to title data (before_render)
- Inject CSS styles (after_render)
"""

import re
from typing import Any, Dict, List, Optional

from loguru import logger

from extensions.base import ReportDataTransform, HTMLRenderHook
from .report_dedupe import transform_report_data


# CSS styles for multi-platform source badges - injected via after_render
# Makes multi-source items visually distinct from single-source items
MULTI_PLATFORM_CSS = """
            /* Multi-platform source badge styles - injected by report_dedupe */
            .source-name .platform-link {
                display: inline-block;
                padding: 2px 8px;
                margin-right: 4px;
                background: #f0f9ff;
                border-radius: 4px;
                font-size: 12px;
                color: #0369a1;
                text-decoration: none;
                transition: all 0.2s ease;
            }

            .source-name .platform-link:hover {
                background: #e0f2fe;
                color: #0284c7;
            }

            .source-name .platform-link:last-child {
                margin-right: 0;
            }
"""


class ReportDedupePlugin(ReportDataTransform, HTMLRenderHook):
    """
    Plugin for deduplicating similar news titles.

    This plugin uses configurable similarity thresholds and optionally
    AI-powered (Ollama) judgment to merge similar titles from different
    platforms into unified entries.

    Also implements HTMLRenderHook to:
    - Single-source items: plain text source name (before_render clears urls)
    - Multi-source items: clickable platform badges with CSS styling
    - Remove " / " separators between platform links (after_render)
    """

    # Class attributes (used as default values before config is applied)
    _name = "report_dedupe"
    _version = "1.2.0"

    def __init__(self):
        self._enabled = True
        self.config: Dict[str, Any] = {}
        # Store URL mappings for post-processing
        # Key: (source_name, title) tuple, Value: list of {url, source} dicts
        self._url_mappings: Dict[tuple, List[Dict[str, str]]] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def enabled(self) -> bool:
        return self._enabled

    def apply_config(self, config: Dict[str, Any]) -> None:
        """
        Apply plugin configuration.

        Config format (from config/extensions/report_dedupe.yaml):
        ```yaml
        enabled: true
        strategy: "auto"
        similarity:
          threshold: 0.85
          max_ai_checks: 50
        ollama:
          base_url: "http://localhost:11434"
          model: "qwen2.5:14b-instruct"
        merge:
          source_separator: " / "
          count_strategy: "sum"
          max_items_per_group: 10
        display:
          show_multi_platform_links: true
          max_links: 5
          highlight_primary: true
        ```
        """
        self.config = config or {}
        self._enabled = self.config.get("enabled", False)

    def transform(
        self,
        report_data: Dict[str, Any],
        config: Dict[str, Any],
        context: Any,
    ) -> Dict[str, Any]:
        """
        Transform report data by deduplicating similar titles.

        Args:
            report_data: Report data dictionary
            config: Plugin configuration
            context: Application context (not used in this plugin)

        Returns:
            Transformed report data with deduplicated titles
        """
        # Merge config: parameter takes precedence over stored config
        merged_config = {**self.config, **config}

        logger.debug("[report_dedupe] Config loaded: {}", merged_config)
        logger.debug("[report_dedupe] Plugin enabled: {}", self.enabled)
        logger.debug(
            "[report_dedupe] Checking enabled flag: {}", merged_config.get("enabled")
        )

        if not merged_config.get("enabled"):
            logger.warning(
                "[report_dedupe] Plugin disabled by config, skipping transform"
            )
            return report_data

        logger.info(
            "[report_dedupe] Starting deduplication on {} stats",
            len(report_data.get("stats", [])),
        )

        # 获取主 AI 客户端供扩展使用
        ai_client = None
        ollama_config = merged_config.get("ollama", {})
        ollama_model = ollama_config.get("model", "")
        if ollama_model and context:
            try:
                from extensions.ai_client import create_ai_client
                ai_client = create_ai_client(
                    context.config,
                    model=ollama_model,
                    thinking=False  # 禁用 thinking 模式
                )
                if ai_client:
                    logger.info("[report_dedupe] Using main AI client with Ollama model: {}", ollama_model)
            except Exception as e:
                logger.debug("[report_dedupe] Could not create AI client: {}", e)

        # 注入 AI 客户端到配置
        merged_config["ai_client"] = ai_client

        result = transform_report_data(report_data, merged_config)

        # Store URL mappings for HTML post-processing
        self._url_mappings.clear()
        total_urls_collected = 0
        for stat in result.get("stats", []):
            for title_data in stat.get("titles", []):
                urls = title_data.get("urls", [])
                total_urls_collected += len(urls)
                if urls and len(urls) > 1:
                    # Store mapping using (source_name, title) as key
                    key = (title_data.get("source_name", ""), title_data.get("title", ""))
                    self._url_mappings[key] = urls
                    logger.debug(
                        "[report_dedupe] Stored URL mapping for '{}': {} URLs from {}",
                        title_data.get("title", "")[:30],
                        len(urls),
                        [u.get("source") for u in urls],
                    )

        logger.info(
            "[report_dedupe] Deduplication complete. Stats count: {}, URL mappings: {}, Total URLs collected: {}",
            len(result.get("stats", [])),
            len(self._url_mappings),
            total_urls_collected,
        )
        return result

    def before_render(
        self,
        report_data: Dict[str, Any],
        config: Dict[str, Any],
        context: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Modify title data for source header rendering control.

        Behavior:
        - Single-URL items: Clear urls so source-name renders as plain text
        - Multi-URL items (merged): Keep urls so source-name shows clickable badges

        Args:
            report_data: Report data dictionary
            config: Plugin configuration
            context: Application context

        Returns:
            Modified report data
        """
        merged_config = {**self.config, **config}
        if not merged_config.get("enabled"):
            logger.debug("[report_dedupe] Plugin disabled, skipping before_render")
            return report_data

        single_url_count = 0
        multi_url_count = 0

        def process_title(title_data: Dict[str, Any]) -> None:
            """Process a single title data item."""
            nonlocal single_url_count, multi_url_count

            urls = title_data.get("urls", [])

            if urls and len(urls) > 1:
                # Multi-URL item: keep urls so upstream adds platform-link badges
                multi_url_count += 1
            elif urls and len(urls) == 1:
                # Single-URL item: clear urls so upstream shows plain text source name
                single_url_count += 1
                title_data["urls"] = []

        # Process stats titles
        for stat in report_data.get("stats", []):
            for title_data in stat.get("titles", []):
                process_title(title_data)

        # Process new_titles
        for source_data in report_data.get("new_titles", []):
            for title_data in source_data.get("titles", []):
                process_title(title_data)

        logger.info(
            "[report_dedupe] before_render: {} single-URL (plain text), {} multi-URL (with badges)",
            single_url_count,
            multi_url_count,
        )

        return report_data

    def after_render(
        self,
        html_content: str,
        config: Dict[str, Any],
        context: Any,
    ) -> str:
        """
        Clean up multi-platform source headers in rendered HTML.

        Removes the " / " separators between platform links in the source-name
        span, replacing with direct concatenation for cleaner appearance.

        Args:
            html_content: Rendered HTML string
            config: Plugin configuration
            context: Application context

        Returns:
            Modified HTML string with cleaned source headers
        """
        merged_config = {**self.config, **config}
        if not merged_config.get("enabled"):
            return html_content

        # Remove " / " separators between platform links in source-name spans
        # Pattern: </a> / <a  →  </a><a
        original_length = len(html_content)
        html_content = html_content.replace('</a> / <a', '</a><a')

        replaced_count = (original_length - len(html_content)) // 3  # Each replacement removes 3 chars " / "

        if replaced_count > 0:
            logger.info(
                "[report_dedupe] after_render: cleaned {} multi-platform source separators",
                replaced_count,
            )

            # Inject CSS styles for multi-platform badges
            if "</style>" in html_content:
                html_content = html_content.replace(
                    "</style>",
                    f"{MULTI_PLATFORM_CSS}</style>",
                    1,
                )
                logger.debug("[report_dedupe] Injected multi-platform badge CSS styles")

        return html_content


# Plugin instance for auto-discovery via entry points
plugin = ReportDedupePlugin

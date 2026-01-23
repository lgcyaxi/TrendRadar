# coding=utf-8
"""
WeWork Compact Notification Extension

Transforms verbose notification messages into a compact format optimized for
WeWork and other channels with strict size limits.

Features:
- Modular handler architecture for different sections:
  - new_titles: New titles with AI-powered summary (dynamic)
  - current_list: All current trending titles
  - standalone: Standalone display section (special platforms/RSS)
- AI-powered cross-news summary (requires AI_ANALYSIS enabled in main config)
- Appends existing AI analysis section when available
- Complete channel takeover: handles mock mode and actual sending

Configuration (config/extensions/wework_compact.yaml):
    enabled: true
    target_channels:
      - wework
    msg_type: text  # or markdown
    sections:
      new_titles:
        enabled: true
        include_ai_summary: true
        include_ai_analysis: true
      current_list:
        enabled: true
      standalone:
        enabled: true
    section_order:
      - new_titles
      - current_list
      - standalone
"""

import re
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from extensions.base import NotificationEnhancer

from .handlers import (
    BaseHandler,
    HandlerResult,
    CurrentListHandler,
    NewTitlesHandler,
    StandaloneHandler,
)


# WeWork API message limit (4096 chars for text, keeping some margin for splitting)
WEWORK_MSG_LIMIT = 2048


def strip_markdown(text: str) -> str:
    """Strip markdown formatting for plain text output."""
    # Remove bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Remove links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    return text


def split_message_by_sections(content: str, separator: str, max_length: int) -> List[str]:
    """
    Split content into multiple messages based on section separators.

    Strategy:
    1. Try to split at section separators first
    2. If a single section is still too long, split at line boundaries
    3. Ensure each chunk is under max_length

    Args:
        content: Full message content
        separator: Section separator string
        max_length: Maximum length per message

    Returns:
        List of message chunks
    """
    if len(content) <= max_length:
        return [content]

    # Split by section separator
    sections = content.split(separator)
    chunks = []
    current_chunk = ""

    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        # Check if adding this section would exceed limit
        test_chunk = current_chunk + (separator if current_chunk else "") + section

        if len(test_chunk) <= max_length:
            current_chunk = test_chunk
        else:
            # Save current chunk if not empty
            if current_chunk:
                chunks.append(current_chunk)

            # If single section is too long, split by lines
            if len(section) > max_length:
                line_chunks = _split_by_lines(section, max_length)
                chunks.extend(line_chunks[:-1])  # Add all but last
                current_chunk = line_chunks[-1] if line_chunks else ""
            else:
                current_chunk = section

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [content[:max_length]]


def _split_by_lines(text: str, max_length: int) -> List[str]:
    """Split text by line boundaries when it's too long."""
    lines = text.split('\n')
    chunks = []
    current_chunk = ""

    for line in lines:
        test_line = current_chunk + ("\n" if current_chunk else "") + line

        if len(test_line) <= max_length:
            current_chunk = test_line
        else:
            if current_chunk:
                chunks.append(current_chunk)

            # If single line is too long, hard split it
            if len(line) > max_length:
                # Split at max_length boundaries
                for j in range(0, len(line), max_length):
                    chunk = line[j:j + max_length]
                    if j + max_length < len(line):
                        chunks.append(chunk)
                    else:
                        current_chunk = chunk
            else:
                current_chunk = line

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [text[:max_length]]


class WeWorkCompactPlugin(NotificationEnhancer):
    """
    Compact notification formatter for WeWork and similar channels.

    Uses a modular handler architecture to process different sections:
    - new_titles: Newly detected titles with AI summary
    - current_list: All current trending titles
    - standalone: Standalone display section
    """

    @property
    def name(self) -> str:
        return "wework_compact"

    @property
    def version(self) -> str:
        return "2.0.0"

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self._enabled = True
        self._target_channels: List[str] = ["wework"]
        self._handlers: Dict[str, BaseHandler] = {}
        self._section_order: List[str] = ["new_titles", "current_list", "standalone"]
        self._section_separator = "\n\n━━━━━━━━━━━━━━\n\n"

    def apply_config(self, config: Dict) -> None:
        """Apply plugin configuration from YAML file."""
        if config is None:
            config = {}
        self.config = config

        try:
            self._enabled = config.get("enabled", True) if config else True
            self._target_channels = config.get("target_channels", ["wework"])
            self._msg_type = config.get("msg_type", "text")

            # Get section order from config or use default
            self._section_order = config.get("section_order", [
                "new_titles",
                "current_list",
                "standalone",
            ])

            # Initialize handlers with their section-specific config
            sections_config = config.get("sections", {})
            self._init_handlers(sections_config, config)

            logger.info(
                "[wework_compact] Loaded plugin v{} (enabled={}, channels={}, msg_type={}, sections={})",
                self.version,
                self._enabled,
                self._target_channels,
                self._msg_type,
                list(self._handlers.keys()),
            )
        except Exception as e:
            logger.error("[wework_compact] Error in apply_config: {}", e)
            self._enabled = False

    def _init_handlers(
        self, sections_config: Dict[str, Any], global_config: Dict[str, Any]
    ) -> None:
        """
        Initialize section handlers based on configuration.

        Args:
            sections_config: Section-specific configuration
            global_config: Global plugin configuration for fallback values
        """
        self._handlers = {}

        # Handler class mapping
        handler_classes = {
            "new_titles": NewTitlesHandler,
            "current_list": CurrentListHandler,
            "standalone": StandaloneHandler,
        }

        # Default section configs (merged with global config for shared settings)
        default_configs = {
            "new_titles": {
                "enabled": True,
                "include_ai_summary": True,
                "max_titles": 20,
                "max_platform_display": 3,
                "ollama": global_config.get("ollama", {}),  # Ollama config from extension config
            },
            "current_list": {
                "enabled": True,
                "max_titles": 50,
                "max_platform_display": 3,
            },
            "standalone": {
                "enabled": True,
                "max_items_per_source": 10,
            },
        }

        logger.info("[wework_compact] Initializing handlers with ollama config: {}", global_config.get("ollama", {}))

        for section_name, handler_class in handler_classes.items():
            # Get section config (merge default with user config)
            section_config = {
                **default_configs.get(section_name, {}),
                **sections_config.get(section_name, {}),
            }

            # Only create handler if enabled
            if section_config.get("enabled", True):
                self._handlers[section_name] = handler_class(section_config)
                logger.debug(
                    "[wework_compact] Initialized handler: {} (config={})",
                    section_name,
                    section_config,
                )

    def enhance(
        self,
        content: str,
        channel: str,
        config: Dict[str, Any],
        context: Any,
    ) -> str:
        """
        Enhance notification content by replacing it with compact format.

        Note: This method is kept for backward compatibility but the main
        entry point is now the send() method which takes over the channel.
        """
        if not self._enabled:
            return content

        if channel not in self._target_channels:
            return content

        try:
            main_config = getattr(context, 'config', {})
            enhanced_content = self._generate_content(context, main_config)
            if enhanced_content:
                return enhanced_content
            return content
        except Exception as e:
            logger.error("[wework_compact] Error enhancing content: {}", e)
            return content

    def send(
        self,
        channel: str,
        config: Dict[str, Any],
        context: Any,
    ) -> Optional[bool]:
        """
        Handle the entire wework send operation.

        This method takes over the wework channel's send process, including:
        - Content generation via modular handlers
        - Mock mode handling (print to console)
        - Actual webhook sending

        Args:
            channel: Notification channel name
            config: Plugin configuration
            context: Application context with report_data, ai_analysis, main config

        Returns:
            True if send succeeded, False if failed, None if not handled
        """
        if not self._enabled:
            return None

        if channel not in self._target_channels:
            return None

        try:
            main_config = getattr(context, 'config', {})
            mock_mode = main_config.get('NOTIFICATION_MOCK_MODE', False)

            # Generate enhanced content using handlers
            enhanced_content = self._generate_content(context, main_config)
            if not enhanced_content:
                logger.debug("[wework_compact] No content generated")
                return None

            # Get msg_type from extension config, fallback to main config
            msg_type = self._msg_type or main_config.get('WEWORK_MSG_TYPE', 'text')
            is_text_mode = msg_type.lower() == 'text'

            # Split content if too long
            display_content = strip_markdown(enhanced_content) if is_text_mode else enhanced_content
            message_chunks = split_message_by_sections(
                display_content, self._section_separator, WEWORK_MSG_LIMIT
            )

            if mock_mode:
                # Mock mode: print to console
                mode_label = "text" if is_text_mode else "markdown"
                total_chunks = len(message_chunks)

                for idx, chunk in enumerate(message_chunks, 1):
                    print("\n" + "=" * 60)
                    if total_chunks > 1:
                        print(f"[企业微信 Mock] 增强后内容 ({mode_label}) [{idx}/{total_chunks}]:")
                    else:
                        print(f"[企业微信 Mock] 增强后内容 ({mode_label}):")
                    print("-" * 60)
                    print(chunk)
                    print("=" * 60 + "\n")

                if total_chunks > 1:
                    print(f"📨 消息已拆分为 {total_chunks} 条发送，总长度: {len(display_content)} 字符")

                return True

            # Actual send (with split support)
            return self._send_to_wework(message_chunks, main_config, is_text_mode)

        except Exception as e:
            logger.error("[wework_compact] Error in send: {}", e)
            return None

    def _generate_content(
        self, context: Any, main_config: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate the enhanced content using all enabled handlers.

        Args:
            context: Application context
            main_config: Main application configuration

        Returns:
            Combined content from all handlers, or None if no content
        """
        results: List[HandlerResult] = []

        # Execute handlers in configured order
        for section_name in self._section_order:
            if section_name not in self._handlers:
                continue

            handler = self._handlers[section_name]
            try:
                result = handler.process(context, main_config)
                if result and result.content:
                    results.append(result)
                    logger.debug(
                        "[wework_compact] Handler {} produced {} items",
                        section_name,
                        result.item_count,
                    )
                elif result and result.error:
                    logger.warning(
                        "[wework_compact] Handler {} failed: {}",
                        section_name,
                        result.error,
                    )
            except Exception as e:
                logger.error(
                    "[wework_compact] Handler {} exception: {}",
                    section_name,
                    e,
                )

        if not results:
            return None

        # Combine results with separator
        content_parts = [r.content for r in results if r.content]
        if not content_parts:
            return None

        return self._section_separator.join(content_parts)

    def _send_to_wework(
        self,
        message_chunks: List[str],
        main_config: Dict[str, Any],
        is_text_mode: bool,
    ) -> bool:
        """
        Send content to WeWork webhook(s).

        Args:
            message_chunks: List of message chunks to send (already split if needed)
            main_config: Main application config
            is_text_mode: True for text mode, False for markdown

        Returns:
            True if all sends succeeded, False otherwise
        """
        webhook_url = main_config.get('WEWORK_WEBHOOK_URL', '')
        if not webhook_url:
            logger.warning("[wework_compact] No WEWORK_WEBHOOK_URL configured")
            return False

        # Parse multiple webhooks (separated by ;)
        webhooks = [w.strip() for w in webhook_url.split(';') if w.strip()]
        max_accounts = main_config.get('MAX_ACCOUNTS_PER_CHANNEL', 3)
        webhooks = webhooks[:max_accounts]

        if not webhooks:
            return False

        proxy_url = main_config.get('PROXY_URL', '')
        proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

        total_chunks = len(message_chunks)
        results = []

        for i, url in enumerate(webhooks):
            account_label = f"账号{i+1}" if len(webhooks) > 1 else ""
            account_success = True

            for chunk_idx, chunk in enumerate(message_chunks, 1):
                try:
                    if is_text_mode:
                        payload = {"msgtype": "text", "text": {"content": chunk}}
                    else:
                        payload = {"msgtype": "markdown", "markdown": {"content": chunk}}

                    resp = requests.post(url, json=payload, proxies=proxies, timeout=30)
                    resp.raise_for_status()

                    if total_chunks > 1:
                        logger.debug(
                            "[wework_compact] Sent chunk {}/{} to webhook{}",
                            chunk_idx, total_chunks, account_label
                        )
                except Exception as e:
                    logger.error(
                        "[wework_compact] Failed to send chunk {}/{} to webhook{}: {}",
                        chunk_idx, total_chunks, account_label, e
                    )
                    account_success = False
                    break  # Stop sending remaining chunks for this account

            if account_success:
                if total_chunks > 1:
                    print(f"✅ 企业微信{account_label} 增强消息发送成功 ({total_chunks} 条)")
                else:
                    print(f"✅ 企业微信{account_label} 增强消息发送成功")
                results.append(True)
            else:
                print(f"❌ 企业微信{account_label} 增强消息发送失败")
                results.append(False)

        return any(results) if results else False


def display_mock_notification(
    config: Dict[str, Any],
    stats: List[Dict],
    report_type: str,
    mode: str,
    failed_ids: Optional[List] = None,
    new_titles: Optional[Dict] = None,
    id_to_name: Optional[Dict] = None,
    rss_items: Optional[List[Dict]] = None,
    rss_new_items: Optional[List[Dict]] = None,
    standalone_data: Optional[Dict] = None,
    ai_result: Optional[Any] = None,
    update_info: Optional[Any] = None,
) -> None:
    """
    Display mock notification content.

    First tries to use wework_compact extension for enhanced compact format,
    then falls back to standard formatters.

    Args:
        config: Main application configuration
        stats: Statistics data for the report
        report_type: Type of report (e.g., 'daily', 'hot')
        mode: Display mode
        failed_ids: List of failed platform IDs
        new_titles: New trending titles data
        id_to_name: Mapping of platform IDs to names
        rss_items: RSS feed items
        rss_new_items: New RSS items
        standalone_data: Standalone data for special platforms
        ai_result: AI analysis result
        update_info: Version update information
    """
    from dataclasses import dataclass
    from extensions import get_extension_manager

    # Try to use wework_compact extension for enhanced format
    try:
        ext_manager = get_extension_manager()

        @dataclass
        class NotificationContext:
            """Context object passed to extension send() method."""
            config: Dict
            report_data: Dict
            ai_analysis: Optional[Any] = None
            standalone_data: Optional[Dict] = None

        # Prepare report_data (convert format if needed)
        rank_threshold = config.get("RANK_THRESHOLD", 10)
        processed_new_titles = []
        if new_titles:
            if isinstance(new_titles, list):
                processed_new_titles = new_titles
            else:
                # Dict format: {source_id: {title: title_data}} - convert to list format
                for source_id, titles_data in new_titles.items():
                    if isinstance(titles_data, dict):
                        source_titles = []
                        for title, title_data in titles_data.items():
                            if isinstance(title_data, dict):
                                # Extract time_display from first_time and last_time
                                first_time = title_data.get("first_time", "")
                                last_time = title_data.get("last_time", "")
                                if first_time and last_time and first_time != last_time:
                                    time_display = f"{first_time}~{last_time}"
                                elif first_time:
                                    time_display = first_time
                                else:
                                    time_display = ""

                                source_titles.append({
                                    "title": title,
                                    "url": title_data.get("url", ""),
                                    "mobile_url": title_data.get("mobileUrl", ""),
                                    "ranks": title_data.get("ranks", []),
                                    "rank_threshold": rank_threshold,
                                    "time_display": time_display,
                                    "count": title_data.get("count", 1),
                                })
                        if source_titles:
                            processed_new_titles.append({
                                "source_id": source_id,
                                "source_name": id_to_name.get(source_id, source_id) if id_to_name else source_id,
                                "titles": source_titles,
                            })

        report_data = {
            "stats": stats,
            "new_titles": processed_new_titles,
            "failed_ids": failed_ids or [],
            "total_new_count": sum(len(t.get("titles", [])) for t in processed_new_titles if isinstance(t, dict)),
        }

        context = NotificationContext(
            config=config,
            report_data=report_data,
            ai_analysis=ai_result,
            standalone_data=standalone_data,
        )

        # Delegate to wework_compact extension's send() method (which handles mock mode)
        result = ext_manager.apply_notification_send("wework", context)
        if result is not None:
            # Extension handled the notification
            return

    except Exception as e:
        logger.debug("[Mock 通知] 扩展处理失败，使用标准格式: {}", e)

    # Fallback: use standard formatter (original logic)
    _display_mock_notification_standard(
        config, stats, report_type, mode, failed_ids, new_titles, id_to_name,
        rss_items, rss_new_items, standalone_data, ai_result, update_info
    )


def _display_mock_notification_standard(
    config: Dict[str, Any],
    stats: List[Dict],
    report_type: str,
    mode: str,
    failed_ids: Optional[List] = None,
    new_titles: Optional[Dict] = None,
    id_to_name: Optional[Dict] = None,
    rss_items: Optional[List[Dict]] = None,
    rss_new_items: Optional[List[Dict]] = None,
    standalone_data: Optional[Dict] = None,
    ai_result: Optional[Any] = None,
    update_info: Optional[Any] = None,
) -> None:
    """
    Display mock notification content using standard formatters.

    This is the fallback when wework_compact extension is not available.
    """
    from trendradar.notification.splitter import split_content_into_batches
    from trendradar.notification.batch import get_max_batch_header_size, add_batch_headers
    from trendradar.notification.formatters import strip_markdown
    from trendradar.notification.senders import _render_ai_analysis

    is_text_mode = config.get("WEWORK_MSG_TYPE", "markdown").lower() == "text"
    header_format_type = "wework_text" if is_text_mode else "wework"
    header_reserve = get_max_batch_header_size(header_format_type)
    batch_size = config.get("MESSAGE_BATCH_SIZE", 4000)

    # Prepare report_data for rendering
    # Convert new_titles from dict format to list format if needed
    rank_threshold = config.get("RANK_THRESHOLD", 10)
    processed_new_titles = []
    if new_titles:
        if isinstance(new_titles, list):
            # Already in list format (from deduplication)
            processed_new_titles = new_titles
        else:
            # Dict format: {source_id: {title: title_data}} - convert to list format
            for source_id, titles_data in new_titles.items():
                if isinstance(titles_data, dict):
                    source_titles = []
                    for title, title_data in titles_data.items():
                        if isinstance(title_data, dict):
                            # Extract time_display from first_time and last_time
                            first_time = title_data.get("first_time", "")
                            last_time = title_data.get("last_time", "")
                            if first_time and last_time and first_time != last_time:
                                time_display = f"{first_time}~{last_time}"
                            elif first_time:
                                time_display = first_time
                            else:
                                time_display = ""

                            source_titles.append({
                                "title": title,
                                "url": title_data.get("url", ""),
                                "mobile_url": title_data.get("mobileUrl", ""),
                                "ranks": title_data.get("ranks", []),
                                "rank_threshold": rank_threshold,
                                "time_display": time_display,
                                "count": title_data.get("count", 1),
                            })
                    if source_titles:
                        processed_new_titles.append({
                            "source_id": source_id,
                            "source_name": id_to_name.get(source_id, source_id) if id_to_name else source_id,
                            "titles": source_titles,
                        })

    report_data = {
        "stats": stats,
        "new_titles": processed_new_titles,
        "failed_ids": failed_ids or [],
        "total_new_count": sum(len(t.get("titles", [])) for t in processed_new_titles if isinstance(t, dict)),
    }

    # Render AI content if available
    ai_content = None
    ai_stats = None
    if ai_result:
        ai_content = _render_ai_analysis(ai_result, "wework")
        if getattr(ai_result, "success", False):
            ai_stats = {
                "total_news": getattr(ai_result, "total_news", 0),
                "analyzed_news": getattr(ai_result, "analyzed_news", 0),
                "max_news_limit": getattr(ai_result, "max_news_limit", 0),
                "hotlist_count": getattr(ai_result, "hotlist_count", 0),
                "rss_count": getattr(ai_result, "rss_count", 0),
            }

    # Get timezone from config
    timezone = config.get("TIMEZONE", "Asia/Shanghai")

    # Display regions
    display_regions = config.get("DISPLAY", {}).get("REGIONS", {})

    # Generate batches
    batches = split_content_into_batches(
        report_data, "wework", update_info if config.get("SHOW_VERSION_UPDATE") else None,
        max_bytes=batch_size - header_reserve, mode=mode,
        rss_items=rss_items if display_regions.get("RSS", True) else None,
        rss_new_items=rss_new_items if display_regions.get("RSS", True) else None,
        ai_content=ai_content,
        standalone_data=standalone_data if display_regions.get("STANDALONE", False) else None,
        ai_stats=ai_stats,
        report_type=report_type,
        timezone=timezone,
        show_new_section=display_regions.get("NEW_ITEMS", True),
        rank_threshold=rank_threshold,
    )
    batches = add_batch_headers(batches, header_format_type, batch_size)

    # Display content
    print("\n" + "=" * 60)
    print("[Mock 通知内容 - 标准格式]")
    print("=" * 60)
    print(f"[消息分为 {len(batches)} 批次]\n")

    for i, batch_content in enumerate(batches, 1):
        if is_text_mode:
            content = strip_markdown(batch_content)
        else:
            content = batch_content

        print(f"--- 批次 {i}/{len(batches)} ---")
        print(content)
        print()

    print("=" * 60 + "\n")


# Plugin instance for entry point discovery
plugin = WeWorkCompactPlugin()

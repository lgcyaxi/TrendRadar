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
                "include_ai_analysis": global_config.get("compact_summary", {}).get("include_ai_analysis", True),
                "max_titles": global_config.get("compact_summary", {}).get("max_titles", 20),
                "max_platform_display": global_config.get("fallback", {}).get("max_platform_display", 3),
                "ai": global_config.get("ai", {}),
            },
            "current_list": {
                "enabled": True,
                "max_titles": 50,
                "max_platform_display": global_config.get("fallback", {}).get("max_platform_display", 3),
            },
            "standalone": {
                "enabled": True,
                "max_items_per_source": 10,
            },
        }

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

            if mock_mode:
                # Mock mode: print to console
                mode_label = "text" if is_text_mode else "markdown"
                display_content = strip_markdown(enhanced_content) if is_text_mode else enhanced_content
                print("\n" + "=" * 60)
                print(f"[企业微信 Mock] 增强后内容 ({mode_label}):")
                print("-" * 60)
                print(display_content)
                print("=" * 60 + "\n")
                return True

            # Actual send
            return self._send_to_wework(enhanced_content, main_config, is_text_mode)

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
        content: str,
        main_config: Dict[str, Any],
        is_text_mode: bool,
    ) -> bool:
        """Send content to WeWork webhook(s)."""
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

        results = []
        for i, url in enumerate(webhooks):
            account_label = f"账号{i+1}" if len(webhooks) > 1 else ""
            try:
                if is_text_mode:
                    plain_content = strip_markdown(content)
                    payload = {"msgtype": "text", "text": {"content": plain_content}}
                else:
                    payload = {"msgtype": "markdown", "markdown": {"content": content}}

                resp = requests.post(url, json=payload, proxies=proxies, timeout=30)
                resp.raise_for_status()
                print(f"✅ 企业微信{account_label} 增强消息发送成功")
                results.append(True)
            except Exception as e:
                print(f"❌ 企业微信{account_label} 增强消息发送失败: {e}")
                results.append(False)

        return any(results) if results else False


# Plugin instance for entry point discovery
plugin = WeWorkCompactPlugin()

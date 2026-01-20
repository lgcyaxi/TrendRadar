# coding=utf-8
"""
Tests for WeWork Compact Extension

Tests the modular handler architecture for wework_compact extension:
- NewTitlesHandler (本日新增)
- CurrentListHandler (当前榜单)
- StandaloneHandler (独立展示区)
- WeWorkCompactPlugin (orchestrator)
"""

import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from extensions.wework_compact import WeWorkCompactPlugin, strip_markdown
from extensions.wework_compact.handlers import (
    BaseHandler,
    HandlerResult,
    CurrentListHandler,
    NewTitlesHandler,
    StandaloneHandler,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@dataclass
class MockContext:
    """Mock context for testing handlers."""
    config: Dict[str, Any]
    report_data: Dict[str, Any]
    ai_analysis: Optional[Any] = None
    standalone_data: Optional[Dict] = None


@pytest.fixture
def basic_config():
    """Basic wework_compact configuration."""
    return {
        "enabled": True,
        "target_channels": ["wework"],
        "msg_type": "text",
        "sections": {
            "new_titles": {
                "enabled": True,
                "include_ai_summary": False,  # Disable AI for unit tests
                "include_ai_analysis": True,
                "max_titles": 20,
                "max_platform_display": 3,
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
        },
        "section_order": ["new_titles", "current_list", "standalone"],
        "compact_summary": {"include_ai_analysis": True, "max_titles": 20},
        "fallback": {"max_platform_display": 3},
        "ai": {"use_main_config": True},
    }


@pytest.fixture
def sample_report_data():
    """Sample report_data with stats and new_titles."""
    return {
        "stats": [
            {
                "keyword": "科技",
                "titles": [
                    {"title": "Apple发布新款iPhone 16", "platforms": ["weibo", "zhihu"]},
                    {"title": "华为Mate70系列震撼发布", "platforms": ["douyin", "toutiao"]},
                    {"title": "特斯拉股价创新高", "platforms": ["baidu"]},
                    {"title": "旧新闻标题不在新增列表", "platforms": ["weibo"]},
                ]
            },
            {
                "keyword": "游戏",
                "titles": [
                    {"title": "Steam秋季特卖开始", "platforms": ["bilibili", "v2ex"]},
                ]
            }
        ],
        "new_titles": [
            {
                "source": "weibo",
                "titles": [
                    {"title": "Apple发布新款iPhone 16"},
                    {"title": "华为Mate70系列震撼发布"},
                ]
            },
            {
                "source": "baidu",
                "titles": [
                    {"title": "特斯拉股价创新高"},
                ]
            },
            {
                "source": "bilibili",
                "titles": [
                    {"title": "Steam秋季特卖开始"},
                ]
            }
        ],
        "id_to_name": {
            "weibo": "微博热搜",
            "zhihu": "知乎热榜",
        }
    }


@pytest.fixture
def sample_standalone_data():
    """Sample standalone_data with platforms and RSS feeds."""
    return {
        "platforms": [
            {
                "id": "weibo",
                "name": "微博热搜",
                "items": [
                    {"title": "微博热搜第一条", "ranks": [1, 2, 3], "url": "https://weibo.com/1"},
                    {"title": "微博热搜第二条", "ranks": [5], "url": "https://weibo.com/2"},
                    {"title": "微博热搜第三条 [详情](https://weibo.com)", "ranks": [10, 8, 7]},
                ]
            },
            {
                "id": "zhihu",
                "name": "知乎热榜",
                "items": [
                    {"title": "知乎问题标题", "ranks": [1]},
                ]
            }
        ],
        "rss_feeds": [
            {
                "id": "36kr",
                "name": "36氪",
                "items": [
                    {"title": "36氪科技新闻", "url": "https://36kr.com/1"},
                    {"title": "创业公司融资消息 https://36kr.com/2"},
                ]
            },
            {
                "id": "ithome",
                "name": "IT之家",
                "items": [
                    {"title": "IT之家快讯"},
                ]
            }
        ]
    }


@pytest.fixture
def report_data_with_urls():
    """Report data with URLs and markdown links to test stripping."""
    return {
        "stats": [
            {
                "keyword": "test",
                "titles": [
                    {"title": "标题带URL https://example.com 后面还有", "platforms": ["weibo"]},
                    {"title": "标题带链接 [点击查看](https://example.com)", "platforms": ["zhihu"]},
                    {"title": "普通标题无URL", "platforms": ["baidu"]},
                ]
            }
        ],
        "new_titles": [
            {
                "source": "weibo",
                "titles": [
                    {"title": "标题带URL  后面还有"},  # Already cleaned in new_titles
                    {"title": "标题带链接 点击查看"},
                    {"title": "普通标题无URL"},
                ]
            }
        ]
    }


# =============================================================================
# HandlerResult Tests
# =============================================================================

class TestHandlerResult:
    """Test HandlerResult dataclass."""

    def test_handler_result_bool_success_with_content(self):
        """HandlerResult should be truthy when success=True and content exists."""
        result = HandlerResult(
            section_name="test",
            title="Test",
            content="Some content",
            item_count=1,
            success=True,
        )
        assert bool(result) is True

    def test_handler_result_bool_success_no_content(self):
        """HandlerResult should be falsy when content is empty."""
        result = HandlerResult(
            section_name="test",
            title="Test",
            content="",
            item_count=0,
            success=True,
        )
        assert bool(result) is False

    def test_handler_result_bool_failure(self):
        """HandlerResult should be falsy when success=False."""
        result = HandlerResult(
            section_name="test",
            title="Test",
            content="Some content",
            item_count=1,
            success=False,
            error="Test error",
        )
        assert bool(result) is False


# =============================================================================
# BaseHandler Tests
# =============================================================================

class TestBaseHandler:
    """Test BaseHandler utility methods."""

    def test_strip_urls(self, basic_config):
        """Test URL stripping from text."""
        handler = CurrentListHandler(basic_config["sections"]["current_list"])

        assert handler._strip_urls("text https://example.com more") == "text  more"
        assert handler._strip_urls("http://test.com start") == "start"
        assert handler._strip_urls("no urls here") == "no urls here"
        assert handler._strip_urls("www.example.com test") == "test"

    def test_strip_markdown_links(self, basic_config):
        """Test markdown link stripping."""
        handler = CurrentListHandler(basic_config["sections"]["current_list"])

        assert handler._strip_markdown_links("[text](http://url.com)") == "text"
        assert handler._strip_markdown_links("before [link](url) after") == "before link after"
        assert handler._strip_markdown_links("no links") == "no links"

    def test_clean_title(self, basic_config):
        """Test combined title cleaning."""
        handler = CurrentListHandler(basic_config["sections"]["current_list"])

        # Should strip both markdown links and URLs
        assert handler._clean_title("[详情](https://example.com)") == "详情"
        assert handler._clean_title("标题 https://url.com") == "标题"
        assert handler._clean_title("标题 [链接](url) 后面 http://test.com") == "标题 链接 后面"

    def test_normalize_platform_names(self, basic_config):
        """Test platform name normalization."""
        handler = CurrentListHandler(basic_config["sections"]["current_list"])

        assert handler._normalize_platform_names(["weibo"]) == ["微博"]
        assert handler._normalize_platform_names(["zhihu", "baidu"]) == ["知乎", "百度"]
        assert handler._normalize_platform_names(["WEIBO"]) == ["微博"]  # Case insensitive
        assert handler._normalize_platform_names(["unknown"]) == ["unknown"]  # Unknown kept
        assert handler._normalize_platform_names([]) == []

    def test_format_title_line(self, basic_config):
        """Test title line formatting."""
        handler = CurrentListHandler(basic_config["sections"]["current_list"])

        assert handler._format_title_line("标题", ["微博", "知乎"], 3) == "- 标题 (微博/知乎)"
        assert handler._format_title_line("标题", [], 3) == "- 标题"
        # Test platform limit
        assert handler._format_title_line("标题", ["a", "b", "c", "d"], 2) == "- 标题 (a/b)"


# =============================================================================
# CurrentListHandler Tests (当前榜单)
# =============================================================================

class TestCurrentListHandler:
    """Test CurrentListHandler (当前榜单)."""

    def test_section_properties(self, basic_config):
        """Test handler section properties."""
        handler = CurrentListHandler(basic_config["sections"]["current_list"])
        assert handler.section_name == "current_list"
        assert handler.section_title == "当前榜单"

    def test_process_with_valid_data(self, basic_config, sample_report_data):
        """Test processing with valid report data."""
        handler = CurrentListHandler(basic_config["sections"]["current_list"])
        context = MockContext(
            config={"AI_ANALYSIS": {"ENABLED": False}},
            report_data=sample_report_data,
        )

        result = handler.process(context, context.config)

        assert result.success is True
        assert result.section_name == "current_list"
        assert result.item_count == 5  # All 5 titles from stats
        assert "当前榜单 (共5条)" in result.content
        assert "Apple发布新款iPhone 16" in result.content
        assert "(微博/知乎)" in result.content  # Platform normalization

    def test_process_with_empty_data(self, basic_config):
        """Test processing with empty report data."""
        handler = CurrentListHandler(basic_config["sections"]["current_list"])
        context = MockContext(
            config={},
            report_data={"stats": []},
        )

        result = handler.process(context, context.config)

        assert result.success is True
        assert result.item_count == 0
        assert result.content == ""

    def test_process_strips_urls(self, basic_config, report_data_with_urls):
        """Test that URLs are stripped from titles."""
        handler = CurrentListHandler(basic_config["sections"]["current_list"])
        context = MockContext(
            config={},
            report_data=report_data_with_urls,
        )

        result = handler.process(context, context.config)

        assert "https://" not in result.content
        assert "[点击查看]" not in result.content
        assert "标题带URL" in result.content

    def test_max_titles_limit(self, basic_config, sample_report_data):
        """Test max_titles configuration limit."""
        config = basic_config["sections"]["current_list"].copy()
        config["max_titles"] = 2
        handler = CurrentListHandler(config)

        context = MockContext(config={}, report_data=sample_report_data)
        result = handler.process(context, context.config)

        assert result.item_count == 2
        assert "当前榜单 (共2条)" in result.content

    def test_process_no_context(self, basic_config):
        """Test processing when context has no report_data."""
        handler = CurrentListHandler(basic_config["sections"]["current_list"])

        result = handler.process(None, {})

        assert result.success is False
        assert "No report_data" in result.error

    def test_process_with_ai_analysis(self, basic_config, sample_report_data):
        """Test that AI analysis is appended to current_list when available."""
        config = basic_config["sections"]["current_list"].copy()
        config["include_ai_analysis"] = True

        handler = CurrentListHandler(config)

        # Mock AI analysis result
        mock_ai = MagicMock()
        mock_ai.success = True
        mock_ai.core_trends = "核心趋势内容"
        mock_ai.sentiment_controversy = "舆论内容"
        mock_ai.signals = ""
        mock_ai.rss_insights = ""
        mock_ai.outlook_strategy = ""

        context = MockContext(
            config={},
            report_data=sample_report_data,
            ai_analysis=mock_ai,
        )

        result = handler.process(context, context.config)

        assert "AI 深度分析" in result.content
        assert "核心趋势内容" in result.content
        assert "舆论内容" in result.content


# =============================================================================
# NewTitlesHandler Tests (本日新增)
# =============================================================================

class TestNewTitlesHandler:
    """Test NewTitlesHandler (本日新增)."""

    def test_section_properties(self, basic_config):
        """Test handler section properties."""
        handler = NewTitlesHandler(basic_config["sections"]["new_titles"])
        assert handler.section_name == "new_titles"
        assert handler.section_title == "本日新增"

    def test_process_with_valid_data(self, basic_config, sample_report_data):
        """Test processing with valid report data."""
        handler = NewTitlesHandler(basic_config["sections"]["new_titles"])
        context = MockContext(
            config={"AI_ANALYSIS": {"ENABLED": False}},
            report_data=sample_report_data,
        )

        result = handler.process(context, context.config)

        assert result.success is True
        assert result.section_name == "new_titles"
        # Should only include titles that are in BOTH stats AND new_titles
        assert result.item_count == 4  # 4 new titles match
        assert "本日新增: 共发现 4 条新热点" in result.content
        assert "Apple发布新款iPhone 16" in result.content
        # Old title should NOT be included
        assert "旧新闻标题不在新增列表" not in result.content

    def test_process_filters_to_new_only(self, basic_config, sample_report_data):
        """Test that only new titles are included."""
        handler = NewTitlesHandler(basic_config["sections"]["new_titles"])
        context = MockContext(
            config={"AI_ANALYSIS": {"ENABLED": False}},
            report_data=sample_report_data,
        )

        result = handler.process(context, context.config)

        # "旧新闻标题不在新增列表" is in stats but NOT in new_titles
        assert "旧新闻标题不在新增列表" not in result.content

    def test_process_with_no_new_titles(self, basic_config):
        """Test processing when there are no new titles."""
        handler = NewTitlesHandler(basic_config["sections"]["new_titles"])
        context = MockContext(
            config={"AI_ANALYSIS": {"ENABLED": False}},
            report_data={
                "stats": [{"keyword": "test", "titles": [{"title": "Old title", "platforms": ["weibo"]}]}],
                "new_titles": [],  # No new titles
            },
        )

        result = handler.process(context, context.config)

        assert result.success is True
        assert result.item_count == 0
        assert result.content == ""

    def test_process_no_ai_analysis_included(self, basic_config, sample_report_data):
        """Test that AI analysis is NOT included in new_titles (moved to current_list)."""
        config = basic_config["sections"]["new_titles"].copy()

        handler = NewTitlesHandler(config)

        # Mock AI analysis result (should be ignored by new_titles handler)
        mock_ai = MagicMock()
        mock_ai.success = True
        mock_ai.core_trends = "核心趋势内容"
        mock_ai.sentiment_controversy = ""
        mock_ai.signals = ""
        mock_ai.rss_insights = ""
        mock_ai.outlook_strategy = ""

        context = MockContext(
            config={"AI_ANALYSIS": {"ENABLED": False}},
            report_data=sample_report_data,
            ai_analysis=mock_ai,
        )

        result = handler.process(context, context.config)

        # AI analysis should NOT be in new_titles (moved to current_list)
        assert "AI 深度分析" not in result.content
        assert "核心趋势内容" not in result.content

    def test_fallback_format_without_ai(self, basic_config, sample_report_data):
        """Test fallback format when AI is disabled."""
        handler = NewTitlesHandler(basic_config["sections"]["new_titles"])
        context = MockContext(
            config={"AI_ANALYSIS": {"ENABLED": False}},
            report_data=sample_report_data,
        )

        result = handler.process(context, context.config)

        # Should use fallback format (no AI summary, just count)
        assert "本日新增: 共发现" in result.content
        assert "条新热点" in result.content


# =============================================================================
# StandaloneHandler Tests (独立展示区)
# =============================================================================

class TestStandaloneHandler:
    """Test StandaloneHandler (独立展示区)."""

    def test_section_properties(self, basic_config):
        """Test handler section properties."""
        handler = StandaloneHandler(basic_config["sections"]["standalone"])
        assert handler.section_name == "standalone"
        assert handler.section_title == "独立展示区"

    def test_process_with_platforms_and_rss(self, basic_config, sample_standalone_data):
        """Test processing with both platforms and RSS feeds."""
        handler = StandaloneHandler(basic_config["sections"]["standalone"])
        context = MockContext(
            config={},
            report_data={},
            standalone_data=sample_standalone_data,
        )

        result = handler.process(context, context.config)

        assert result.success is True
        assert result.section_name == "standalone"
        assert "独立展示区" in result.content

        # Check platforms
        assert "微博热搜" in result.content
        assert "微博热搜第一条" in result.content
        assert "排名1-3" in result.content  # Rank range format
        assert "知乎热榜" in result.content

        # Check RSS feeds
        assert "36氪" in result.content
        assert "36氪科技新闻" in result.content
        assert "IT之家" in result.content

    def test_process_strips_urls_from_standalone(self, basic_config, sample_standalone_data):
        """Test that URLs are stripped from standalone data."""
        handler = StandaloneHandler(basic_config["sections"]["standalone"])
        context = MockContext(
            config={},
            report_data={},
            standalone_data=sample_standalone_data,
        )

        result = handler.process(context, context.config)

        # URLs and markdown links should be stripped
        assert "https://" not in result.content
        assert "[详情]" not in result.content

    def test_process_with_no_standalone_data(self, basic_config):
        """Test processing when standalone_data is None."""
        handler = StandaloneHandler(basic_config["sections"]["standalone"])
        context = MockContext(
            config={},
            report_data={},
            standalone_data=None,
        )

        result = handler.process(context, context.config)

        assert result.success is True
        assert result.item_count == 0
        assert result.content == ""

    def test_process_with_empty_standalone_data(self, basic_config):
        """Test processing when standalone_data is empty."""
        handler = StandaloneHandler(basic_config["sections"]["standalone"])
        context = MockContext(
            config={},
            report_data={},
            standalone_data={"platforms": [], "rss_feeds": []},
        )

        result = handler.process(context, context.config)

        assert result.success is True
        assert result.item_count == 0
        assert result.content == ""

    def test_rank_format_single(self, basic_config):
        """Test rank formatting for single rank."""
        handler = StandaloneHandler(basic_config["sections"]["standalone"])

        assert handler._format_rank_range([5]) == "排名5"
        assert handler._format_rank_range([1]) == "排名1"

    def test_rank_format_range(self, basic_config):
        """Test rank formatting for rank range."""
        handler = StandaloneHandler(basic_config["sections"]["standalone"])

        assert handler._format_rank_range([1, 2, 3]) == "排名1-3"
        assert handler._format_rank_range([5, 3, 1]) == "排名1-5"  # Should sort
        assert handler._format_rank_range([10, 8, 7]) == "排名7-10"

    def test_rank_format_empty(self, basic_config):
        """Test rank formatting for empty ranks."""
        handler = StandaloneHandler(basic_config["sections"]["standalone"])

        assert handler._format_rank_range([]) == ""
        assert handler._format_rank_range([0, -1]) == ""  # Invalid ranks

    def test_platform_emoji(self, basic_config):
        """Test platform emoji mapping."""
        handler = StandaloneHandler(basic_config["sections"]["standalone"])

        assert handler._get_platform_emoji("weibo") == "📊"
        assert handler._get_platform_emoji("zhihu") == "💬"
        assert handler._get_platform_emoji("WEIBO") == "📊"  # Case insensitive
        assert handler._get_platform_emoji("unknown") == "📊"  # Default


# =============================================================================
# WeWorkCompactPlugin Integration Tests
# =============================================================================

class TestWeWorkCompactPlugin:
    """Test WeWorkCompactPlugin orchestrator."""

    def test_plugin_initialization(self, basic_config):
        """Test plugin initialization with config."""
        plugin = WeWorkCompactPlugin()
        plugin.apply_config(basic_config)

        assert plugin.name == "wework_compact"
        assert plugin.version == "2.0.0"
        assert plugin._enabled is True
        assert "wework" in plugin._target_channels
        assert "new_titles" in plugin._handlers
        assert "current_list" in plugin._handlers
        assert "standalone" in plugin._handlers

    def test_plugin_section_order(self, basic_config):
        """Test section order configuration."""
        config = basic_config.copy()
        config["section_order"] = ["standalone", "new_titles", "current_list"]

        plugin = WeWorkCompactPlugin()
        plugin.apply_config(config)

        assert plugin._section_order == ["standalone", "new_titles", "current_list"]

    def test_plugin_disabled_section(self, basic_config):
        """Test that disabled sections are not created."""
        config = basic_config.copy()
        config["sections"]["current_list"]["enabled"] = False

        plugin = WeWorkCompactPlugin()
        plugin.apply_config(config)

        assert "current_list" not in plugin._handlers
        assert "new_titles" in plugin._handlers
        assert "standalone" in plugin._handlers

    def test_generate_content_all_sections(self, basic_config, sample_report_data, sample_standalone_data):
        """Test full content generation with all sections."""
        plugin = WeWorkCompactPlugin()
        plugin.apply_config(basic_config)

        context = MockContext(
            config={"AI_ANALYSIS": {"ENABLED": False}},
            report_data=sample_report_data,
            standalone_data=sample_standalone_data,
        )

        content = plugin._generate_content(context, context.config)

        assert content is not None

        # Check all sections present
        assert "本日新增" in content
        assert "当前榜单" in content
        assert "独立展示区" in content

        # Check separator
        assert "━━━━━━━━━━━━━━" in content

    def test_generate_content_partial_data(self, basic_config, sample_report_data):
        """Test content generation with partial data (no standalone)."""
        plugin = WeWorkCompactPlugin()
        plugin.apply_config(basic_config)

        context = MockContext(
            config={"AI_ANALYSIS": {"ENABLED": False}},
            report_data=sample_report_data,
            standalone_data=None,  # No standalone data
        )

        content = plugin._generate_content(context, context.config)

        assert content is not None
        assert "本日新增" in content
        assert "当前榜单" in content
        # Standalone should not appear (no data)
        assert "独立展示区" not in content

    def test_generate_content_no_data(self, basic_config):
        """Test content generation with no data."""
        plugin = WeWorkCompactPlugin()
        plugin.apply_config(basic_config)

        context = MockContext(
            config={"AI_ANALYSIS": {"ENABLED": False}},
            report_data={"stats": [], "new_titles": []},
            standalone_data=None,
        )

        content = plugin._generate_content(context, context.config)

        assert content is None

    def test_send_mock_mode(self, basic_config, sample_report_data, capsys):
        """Test send in mock mode prints to console."""
        plugin = WeWorkCompactPlugin()
        plugin.apply_config(basic_config)

        context = MockContext(
            config={
                "NOTIFICATION_MOCK_MODE": True,
                "AI_ANALYSIS": {"ENABLED": False},
            },
            report_data=sample_report_data,
        )

        result = plugin.send("wework", basic_config, context)

        assert result is True

        # Check console output
        captured = capsys.readouterr()
        assert "[企业微信 Mock]" in captured.out
        assert "本日新增" in captured.out

    def test_send_non_target_channel(self, basic_config, sample_report_data):
        """Test send returns None for non-target channels."""
        plugin = WeWorkCompactPlugin()
        plugin.apply_config(basic_config)

        context = MockContext(
            config={"AI_ANALYSIS": {"ENABLED": False}},
            report_data=sample_report_data,
        )

        result = plugin.send("feishu", basic_config, context)

        assert result is None

    def test_send_disabled_plugin(self, basic_config, sample_report_data):
        """Test send returns None when plugin is disabled."""
        config = basic_config.copy()
        config["enabled"] = False

        plugin = WeWorkCompactPlugin()
        plugin.apply_config(config)

        context = MockContext(
            config={"AI_ANALYSIS": {"ENABLED": False}},
            report_data=sample_report_data,
        )

        result = plugin.send("wework", config, context)

        assert result is None


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilityFunctions:
    """Test utility functions."""

    def test_strip_markdown(self):
        """Test strip_markdown function."""
        assert strip_markdown("**bold**") == "bold"
        assert strip_markdown("*italic*") == "italic"
        assert strip_markdown("__bold__") == "bold"
        assert strip_markdown("_italic_") == "italic"
        assert strip_markdown("[link](url)") == "link"
        assert strip_markdown("`code`") == "code"
        assert strip_markdown("normal text") == "normal text"

    def test_strip_markdown_combined(self):
        """Test strip_markdown with multiple formats."""
        text = "**Bold** and *italic* with [link](url) and `code`"
        result = strip_markdown(text)
        assert result == "Bold and italic with link and code"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handler_with_none_context(self, basic_config):
        """Test handlers handle None context gracefully."""
        handler = CurrentListHandler(basic_config["sections"]["current_list"])

        result = handler.process(None, {})

        assert result.success is False
        assert result.error is not None

    def test_handler_with_malformed_data(self, basic_config):
        """Test handlers handle malformed data gracefully."""
        handler = CurrentListHandler(basic_config["sections"]["current_list"])
        context = MockContext(
            config={},
            report_data={
                "stats": [
                    {"titles": "not a list"},  # Malformed
                    {"keyword": "test"},  # Missing titles
                    None,  # None item - this will cause an exception
                ]
            },
        )

        result = handler.process(context, context.config)

        # Handler returns success=False when encountering malformed data
        # This is acceptable behavior - it doesn't crash and reports the error
        assert result.success is False
        assert result.error is not None

    def test_duplicate_title_deduplication(self, basic_config):
        """Test that duplicate titles are deduplicated."""
        handler = CurrentListHandler(basic_config["sections"]["current_list"])
        context = MockContext(
            config={},
            report_data={
                "stats": [
                    {
                        "keyword": "test",
                        "titles": [
                            {"title": "Same Title", "platforms": ["weibo"]},
                            {"title": "Same Title", "platforms": ["zhihu"]},  # Duplicate
                            {"title": "Different Title", "platforms": ["baidu"]},
                        ]
                    }
                ]
            },
        )

        result = handler.process(context, context.config)

        # Should deduplicate to 2 unique titles
        assert result.item_count == 2
        # Count occurrences of "Same Title" in content
        assert result.content.count("Same Title") == 1

    def test_empty_platforms_handling(self, basic_config):
        """Test handling of titles with no platforms."""
        handler = CurrentListHandler(basic_config["sections"]["current_list"])
        context = MockContext(
            config={},
            report_data={
                "stats": [
                    {
                        "keyword": "test",
                        "titles": [
                            {"title": "Title with platforms", "platforms": ["weibo"]},
                            {"title": "Title without platforms", "platforms": []},
                            {"title": "Title with source", "source_name": "custom"},
                        ]
                    }
                ]
            },
        )

        result = handler.process(context, context.config)

        assert "Title with platforms (微博)" in result.content
        assert "- Title without platforms" in result.content
        assert "- Title with source (custom)" in result.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

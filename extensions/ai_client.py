# coding=utf-8
"""
AI Client Helper for Extensions

Provides a simple interface for extensions to use the main AI client
with model override and thinking mode control.
"""

from typing import Any, Dict, Optional


def create_ai_client(config: Dict[str, Any], model: str = None, thinking: bool = False) -> Optional[Any]:
    """
    为扩展创建 AI 客户端

    Args:
        config: 主配置字典（包含 AI 配置）
        model: Ollama 模型名（如 "qwen2.5:14b-instruct"）
               如果为 None，使用主 AI 配置的模型
        thinking: 是否启用 thinking 模式（GLM 需要禁用）

    Returns:
        AIClient 实例，如果配置无效则返回 None
    """
    ai_config = config.get("AI", {})

    # 检查是否有有效的 AI 配置
    api_key = ai_config.get("API_KEY", "").strip()
    api_base = ai_config.get("API_BASE", "").strip()

    # 如果没有配置 API Key 或 API Base，可能无法正常工作
    if not api_key and not api_base:
        return None

    # 覆盖模型（用于扩展指定不同模型）
    if model:
        # 确保模型前缀正确
        if not model.startswith("ollama/") and "/" not in model:
            model = f"ollama/{model}"
        ai_config = ai_config.copy()
        ai_config["MODEL"] = model

    # 添加禁用 thinking 的参数（通过 extra_params）
    if not thinking:
        ai_config = ai_config.copy()
        if "EXTRA_PARAMS" not in ai_config:
            ai_config["EXTRA_PARAMS"] = {}
        ai_config["EXTRA_PARAMS"]["thinking"] = False

    from trendradar.ai.client import AIClient
    return AIClient(ai_config)

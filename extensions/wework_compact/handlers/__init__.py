# coding=utf-8
"""
WeWork Compact Handlers

Section handlers for the wework_compact extension.
"""

from .base import BaseHandler, HandlerResult
from .current_list import CurrentListHandler
from .new_titles import NewTitlesHandler
from .standalone import StandaloneHandler

__all__ = [
    "BaseHandler",
    "HandlerResult",
    "CurrentListHandler",
    "NewTitlesHandler",
    "StandaloneHandler",
]

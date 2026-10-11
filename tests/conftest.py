"""Test bootstrap for a project that has no SearXNG dependency.

``searxng_bm25_reranker`` imports ``searx.plugins`` at module import time.
SearXNG is deliberately not a runtime dependency, so outside a SearXNG install
those imports are unresolvable. This stubs the minimum surface needed to
import and unit-test the plugin on its own.
"""

from __future__ import annotations

import sys
import types


def _install_searx_stub() -> None:
    if "searx.plugins" in sys.modules:
        return

    class Plugin:
        def __init__(self, plg_cfg: object = None) -> None:
            self.plg_cfg = plg_cfg

    class PluginInfo:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    searx = types.ModuleType("searx")
    plugins = types.ModuleType("searx.plugins")
    plugins.Plugin = Plugin
    plugins.PluginInfo = PluginInfo
    plugins.PluginCfg = type("PluginCfg", (), {})
    searx.plugins = plugins

    sys.modules["searx"] = searx
    sys.modules["searx.plugins"] = plugins


_install_searx_stub()

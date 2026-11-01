"""Test bootstrap for a project that has no SearXNG dependency.

``searxng_bm25_reranker`` imports ``searx`` and ``searx.plugins`` at module
import time. SearXNG is deliberately not a runtime dependency, so outside a
SearXNG install those imports are unresolvable. This stubs the minimum surface
needed to import and unit-test the plugin on its own.

The stub mirrors upstream behaviour in the spots that matter:

* ``PluginCfg`` is a dataclass with a single field, ``active`` -- passing any
  other keyword to it raises ``TypeError``, exactly as ``PluginCfg(**plg_cfg)``
  does in ``PluginStorage.load_settings``.
* ``Plugin.__init__`` copies ``plg_cfg.__dict__`` onto the instance and
  discards the object -- nothing keeps a reference, and ``self.plg_cfg`` does
  not exist.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass


def _install_searx_stub() -> None:
    if "searx.plugins" in sys.modules:
        return

    @dataclass
    class PluginCfg:
        active: bool = False

    class Plugin:
        def __init__(self, plg_cfg: PluginCfg | None = None) -> None:
            if plg_cfg is not None:
                for name, value in vars(plg_cfg).items():
                    setattr(self, name, value)

    class PluginInfo:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    searx = types.ModuleType("searx")
    # Upstream's `searx.settings` is the parsed settings.yml dict. Plugin
    # tunables live in a top-level block keyed by the plugin id.
    searx.settings = {}
    plugins = types.ModuleType("searx.plugins")
    plugins.Plugin = Plugin
    plugins.PluginInfo = PluginInfo
    plugins.PluginCfg = PluginCfg
    searx.plugins = plugins

    sys.modules["searx"] = searx
    sys.modules["searx.plugins"] = plugins


_install_searx_stub()

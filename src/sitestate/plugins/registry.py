"""Plug-in registry: discovery plus suitability checking.

The platform can determine whether the currently subscribed sensor
configuration is suitable for a requested output by comparing the data
types sensors declare against what each processing plug-in consumes.
"""

from __future__ import annotations

from typing import Iterable

from .base import OutputAdapter, ProcessingPlugin, SensorAdapter


class PluginRegistry:
    def __init__(self) -> None:
        self._processors: dict[str, ProcessingPlugin] = {}
        self._outputs: dict[str, OutputAdapter] = {}

    # -- processing plug-ins ----------------------------------------------
    def register_processor(self, plugin: ProcessingPlugin) -> None:
        self._processors[plugin.manifest.name] = plugin

    def processor(self, name: str) -> ProcessingPlugin:
        if name not in self._processors:
            raise KeyError(
                f"no processing plug-in named {name!r}; registered: {sorted(self._processors)}"
            )
        return self._processors[name]

    def processors(self) -> list[ProcessingPlugin]:
        return list(self._processors.values())

    def missing_inputs(self, name: str, available: Iterable[str]) -> list[str]:
        """Which of the plug-in's declared inputs are not available."""
        have = set(available)
        return [c for c in self.processor(name).manifest.consumes if c not in have]

    def suitable_processors(self, available: Iterable[str]) -> list[str]:
        have = set(available)
        return [
            p.manifest.name
            for p in self._processors.values()
            if set(p.manifest.consumes) <= have
        ]

    # -- output adapters ---------------------------------------------------
    def register_output(self, adapter: OutputAdapter) -> None:
        self._outputs[adapter.name] = adapter

    def output(self, name: str) -> OutputAdapter:
        if name not in self._outputs:
            raise KeyError(
                f"no output adapter named {name!r}; registered: {sorted(self._outputs)}"
            )
        return self._outputs[name]

    def outputs(self) -> list[OutputAdapter]:
        return list(self._outputs.values())

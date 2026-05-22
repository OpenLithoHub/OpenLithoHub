"""Reference baselines for hotspot detection and related tasks.

This package collects standalone baseline implementations that are not
core models — they are reproducible reference points cited from the
literature, kept here so users can import them by name without pulling
in a training framework.

Currently exposes:

- :func:`hotspot_batchal.batch_active_select` — Yang2020 §III batch
  active sampling for hotspot pattern selection.
"""

from __future__ import annotations

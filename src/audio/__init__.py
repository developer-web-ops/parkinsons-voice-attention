"""Phase 2 raw-audio Parkinson's pipeline (MDVR-KCL, eGeMAPS, calibrated ML).

Kept deliberately separate from the Phase 1 tabular UCI pipeline that lives in
``src`` proper; nothing here touches ``artifacts/``, ``reports/metrics.json`` or
the deployed 753-feature models.

Importing this package (or any module in it) has **no side effects**: it never
reaches the network and never downloads data. Acquiring the MDVR-KCL corpus is
an explicit command — see :mod:`src.audio.fetch_mdvr_kcl`.
"""

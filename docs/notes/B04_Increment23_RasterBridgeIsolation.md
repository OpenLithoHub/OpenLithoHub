# B04 Increment 23 — Raster Bridge Isolation

Current HEAD already contains Increment 22 semantic hardening:
physical-instance identity, GDS/OASIS member tests, PATH normalization tests,
one-way enclosure and public Sky130 benchmark code.

Increment 23 therefore adds only the missing dense-raster bridge layer.

The generated 512x512 fixture has an independent exact occupied area of
90032 pixels. The earlier Mac return reported 96258 occupied pixels from the
canonical dense loader. Increment 23 measures the full false-positive /
false-negative pattern and the smallest mutual Chebyshev dilation radius.

A finite-grid mask enclosure is not yet a continuous contour, EPE, CD,
Hopkins/SOCS or process-window certificate.

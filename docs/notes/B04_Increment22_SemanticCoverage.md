# B04 Increment 22 — Semantic Coverage Matrix

| Semantic case | Current target |
|---|---|
| Polygon / box | PASS candidate |
| Holes | PASS candidate |
| Layer / datatype | PASS candidate |
| Hierarchy | PASS candidate |
| SREF / ordinary instance | PASS candidate |
| GDS AREF / regular array | distinct member identity required |
| OASIS repetition | tested independently from GDS AREF |
| 90° rotations | exact-grid only |
| Reflection | exact-grid only |
| Magnification | accepted only if transformed geometry remains exact-grid |
| Arbitrary-angle transform | rejected / INCONCLUSIVE when off-grid |
| PATH width + begin/end extensions | parser regression required |
| PATH round ends | parser regression required |
| Join semantics | OPEN until parser regression is stable |
| Coordinate quantization | explicit; no silent theorem-path rounding |
| Periodic fixture | separate semantic model |
| Finite physical layout | real context + certified optical tail required |

Claim firewall: do not claim full-chip streaming, practical speedup, or certified
process-window verification on real GDS/OASIS yet.

Acceptance rule:
`reference_interval +/- independently_certified_bridge_error`
must be contained in the theorem-facing certified interval.

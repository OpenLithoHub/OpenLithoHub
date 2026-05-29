# Patent Analysis: Helmholtz PDE Filter for ILT MRC Compliance

> **Status**: NOT RECOMMENDED — prior art review by external expert found this to be non-novel

## Conclusion

**Do NOT file.** External expert prior art search (2026-05-29) identified substantial prior art that was missed in the initial assessment.

## Key Prior Art (from expert review)

1. **Helmholtz PDE filters are standard in nanophotonics topology optimization** — not just CFD/structural TO:
   - Hassan et al. "Algorithmic Design of Nanophotonic Structure" (Helmholtz filter + threshold projection + continuous adjoint)
   - Sigmund group "Topology optimization methods for inverse design of nano-photonic systems"
   - Stanford SPINS, MIT MEEP, Lumerical, Tidy3D all support PDE-based density filtering

2. **ILT MRC constraint patents are densely filed** — US7716627, US8141004, US7856612, US8689148, US9268900, US8321819 (Luminescent/Synopsys), US10656530 (FreeForm MRC), DiffOPC (ICCAD 2024), FD-ILT (2025)

3. **Topology optimization ↔ ILT cross-domain is not novel** — Synopsys has used level-set TO for ILT since 2007

4. **Non-obviousness is weak** — Replacing density filter with Helmholtz PDE filter is textbook-level substitution for a PHOSITA trained in TO

## Decision

All files associated with this innovation are cleared for push.

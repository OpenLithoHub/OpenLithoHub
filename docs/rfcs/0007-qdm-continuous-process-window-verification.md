# RFC 0007 — Proof-Carrying Continuous Process-Window Verification

**Status:** B04 MVP-1 integration draft

## Scope

This is an opt-in theorem-facing layer. It does **not** replace existing
`epe_max_nm` or `pvband_max_nm`, does not change leaderboard semantics, and is
not foundry sign-off.

The first validated target is the pinned OpenLithoHub commit
`348fa5d86d5355465af98e2c4ce3deac60081a4c`, a public ArF Hopkins square
fixture, and the continuous focus interval `[-10, 10] nm`.

## Provenance

Every dependency has one closed proof level:

- `HEURISTIC`
- `NUMERICAL-DIAGNOSTIC`
- `INTERVAL-CERTIFIED`
- `IMPORTED-QDM-CERTIFIED`

Only the final two levels may discharge a theorem dependency. String matching
is forbidden.

## Status semantics

`PASS` requires all of:

1. every required dependency is certifying and satisfied;
2. `coverage_status == ALL_COMPONENTS_COVERED`;
3. the theorem-facing continuous upper bound is at or below the requested
   tolerance.

An upper bound above tolerance is **not** a FAIL. `FAIL` requires an
independently certified violation lower bound above tolerance. Otherwise the
result is `INCONCLUSIVE`.

In particular:

```text
PARTIAL_COVER != PASS
```

## Spatial completeness

Terminal spatial cells are classified as:

- `SIGN_DEFINITE_EMPTY`
- `CURVATURE_CERTIFIED_EMPTY`
- `SEEDED`
- `SUBDIVIDE`
- `INCONCLUSIVE`

The curvature scale is `2*kappa/H`. A root-free cell whose diameter is below
that scale cannot contain a complete hidden closed threshold component.
Boundary root absence/presence must itself be interval-certified.

## MVP-1 proof result

For the frozen B04 proof-mode contract:

```text
focus interval                         [-10, 10] nm
full-source focus intensity upper      0.002916392
uniform K=24 <-> K=37 intensity upper  0.002836179687594228
combined nominal intensity radius      0.005752571687594228
expanded-band active cells             164
expanded-band kappa lower              0.005891396342472183 / nm
continuous contour Hausdorff upper     0.976436035396712 nm
pairwise focus-contour diameter upper  1.952872070793424 nm
```

The continuous metrics should be exposed under names such as
`continuous_focus_contour_upper_nm` and
`continuous_focus_pairwise_diameter_upper_nm`.

## Proof artifact policy

Normal CI tests the certificate logic and interval state machine. Large proof
blobs are external artifacts bound by SHA-256. If mounted, the replay test
recomputes the final expanded-band gate from the outward-rounded grid.

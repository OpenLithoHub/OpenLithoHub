# References

OpenLithoHub maintains a snapshotted bibliography at
[`docs/references.bib`](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/docs/references.bib)
covering every paper, dataset, and standard that the codebase, leaderboard
schema, and simulator parameters cite.

The keys in that file are the canonical handles used elsewhere in this
documentation and in inline source comments. For example, when you see

> "*per `Yang2023_LithoBench` §3.2*"

in `src/openlithohub/simulators/hopkins_sim.py`, the matching BibTeX entry in
`references.bib` records the full citation, arXiv ID, DOI, and a short note on
*why* that paper is cited.

## Why it's snapshotted

Per the project's reproducibility playbook (§5.2), citations are vendored into
the repository rather than fetched live from arXiv or Crossref:

- Build remains deterministic offline (`mkdocs build --strict` does not depend
  on a remote bibliography).
- Future maintainers can audit when each citation was added (via `git log`)
  without trusting upstream metadata that may have been edited.
- The `note = {...}` field on each entry explains what OpenLithoHub uses the
  paper for, which is useful when a code reference is removed and we need to
  decide whether to retire the citation.

## Adding a new citation

1. Add the entry to `docs/references.bib` using the
   `FirstAuthor<YEAR>_ShortTopic` key style (e.g. `Yang2023_LithoBench`).
2. Include a `note = {...}` summarising why OpenLithoHub cites the paper.
3. When you reference the key from code or docs, use the same string verbatim
   so a future `grep` finds both sides.

## See also

- [Architecture overview](architecture.md)
- [Benchmarks](benchmarks.md)
- [Leaderboard submission guide](leaderboard-submission.md) — points at the
  metric definitions backed by `Yang2023_LithoBench` Table III.

# RFC lifecycle

OpenLithoHub uses a two-stage workflow for design proposals: discussion
happens in an issue, the resulting design lands as a markdown file via PR.

This page documents the workflow so future RFCs follow the same pattern.

## Why two stages

A short look at the precedent: RFC 0006 (MCP Bridge) had its design
discussion at issue [#11](https://github.com/OpenLithoHub/OpenLithoHub/issues/11)
(12 substantive comments from three external contributors before the
design froze) and was then captured as
[`docs/rfcs/0006-mcp-bridge.md`](0006-mcp-bridge.md) via PR
[#19](https://github.com/OpenLithoHub/OpenLithoHub/pull/19).

The split exists because issues and PRs serve different audiences:

- **Issues** are the project's discussion surface. They land on the
  default tracker tab, get notification weight from watchers, and don't
  require the reader to parse a diff. People who care about a feature
  but don't intend to write code engage there.
- **PRs** are the merge surface. They're optimised for "is this change
  safe to integrate" — file diffs, CI status, line-level review. Folding
  open-ended design discussion into a PR pushes non-coders away and
  fragments the conversation across review threads and PR comments.

For RFCs the project wants both: visible discussion that anyone can join,
and a frozen artifact that ships in `docs/rfcs/`. So we use both venues
in sequence.

## Stage 1 — Discussion issue

Open an issue with:

- A title prefixed with the RFC number, e.g.
  `[RFC 0007] Streaming hotspot annotations`. The number is whatever the
  next free `docs/rfcs/NNNN-*.md` slot is (currently `0007`).
- The label `rfc-discussion`.
- A body that covers:
  - **Motivation** — what problem this solves, who is asking, what's
    happening today that's broken or missing.
  - **Proposed design** — the shape of the solution at the level of
    interfaces, file boundaries, and acceptance gates. Detail enough
    that a reader can disagree with specifics, not just direction.
  - **Alternatives considered** — at least one. If you only see one
    path the design is probably not yet ready to write down.
  - **Open questions** — explicit list. This is what comments will
    converge on.

Discuss freely. Edit the issue body to incorporate consensus as it
forms — comments are append-only but the body is the canonical
description and should reflect the latest state.

## Stage 2 — RFC PR

When the open-questions list is empty (or every remaining question is
flagged as a follow-up), open a PR that adds
`docs/rfcs/NNNN-<short-slug>.md`. The file structure mirrors the issue
body but is the *final* form, not a discussion artifact.

The PR body should:

1. Open with `Captures the frozen design from #N as RFC NNNN.` so the
   linkage is visible at the top of the PR.
2. Contain a short changelog of decisions ("frozen gates", "schema
   decisions", etc.) — not the full RFC text, since reviewers can read
   the file in the diff.
3. Acknowledge external contributors from the issue thread by GitHub
   handle, citing which decision came from whom. (See PR #19 for the
   pattern.)
4. State what happens to the issue: typically `Closes #N` once the PR
   merges, on the explicit understanding that further refinement
   happens on the PR itself, not by re-opening the issue.

The PR is for **wording, scope, and consistency review** — not for
re-litigating the design. If a substantive design question surfaces
during PR review, push back on the question to the issue
(re-open #N if necessary) rather than absorbing it into the PR thread.

## When to skip the issue

A short list of cases where opening an issue first is overhead:

- The change is editorial — fixing typos in an existing RFC, renumbering,
  reformatting.
- The RFC is a near-mechanical extension of an already-merged RFC and
  the author is the same person.
- A maintainer asks for a follow-up RFC in a thread and the scope is
  already pinned in that thread.

In all other cases, default to the two-stage flow. The cost of an
extra issue is low; the cost of a stalled discussion buried in a draft
PR is high.

## File-naming and numbering

- Filename: `docs/rfcs/NNNN-<short-slug>.md`. `NNNN` is zero-padded to
  four digits and increments monotonically. `<short-slug>` is
  hyphen-separated lowercase, no leading article.
- Examples that follow the convention: `0004-multi-gpu-tile-pipeline.md`,
  `0006-mcp-bridge.md`.
- Update the `RFCs` section of [`mkdocs.yml`](https://github.com/OpenLithoHub/OpenLithoHub/blob/main/mkdocs.yml)
  with a nav entry: `"NNNN — Title": rfcs/NNNN-slug.md`.

## Status field

Each RFC file should carry a status near the top: `Draft`, `Accepted`, or
`Superseded by NNNN`. Update it on merge and on supersession. The
default status of a freshly merged RFC is `Accepted` — the design is
frozen, not under reconsideration. Use `Draft` only while the file is
still in a PR.

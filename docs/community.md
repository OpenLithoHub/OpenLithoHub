# Community

> **Status — 2026-Q2:** the channel is still being chartered. The Discord
> invite is not yet live. This page documents the structure, etiquette,
> and onboarding flow we are committing to **before** opening the doors,
> so that the first hundred members find a deliberately-designed space
> rather than a generic chatroom.
>
> To be notified when the invite goes live, [open an issue with the
> `community` label](https://github.com/OpenLithoHub/OpenLithoHub/issues/new?labels=community&title=Community+launch+notification)
> or watch the repository.

## Discord

We are running a **single global Discord server**, English-first but
multilingual-friendly. Computational lithography is a small enough field
that fragmenting the conversation across regional channels would hurt
more than help.

### Channel layout

| Channel               | Purpose                                                       |
| --------------------- | ------------------------------------------------------------- |
| `#announcements`      | Read-only. Releases, leaderboard updates, paper milestones.   |
| `#model-discussion`   | Architecture, training, fine-tuning, ablations.               |
| `#physics-simulation` | Hopkins/SOCS, EUV 3D-mask, resist models, MRC, OPC algorithms.|
| `#datasets`           | LithoBench, ICCAD16, GAN-OPC, synthetic generation.           |
| `#help`               | Setup, install, CLI, BYOM. Beginner-friendly.                 |
| `#showcase`           | Show your runs, plots, leaderboard entries.                   |
| `#meta`               | Community feedback, moderation, off-topic.                    |

### Etiquette

- **Reproducibility before claims.** When posting a result, link the
  commit, dataset version, and `openlithohub --version`. "It works on my
  machine" without an artifact gets a polite nudge to attach one.
- **No closed-source tool screenshots without context.** If you compare
  against Calibre/Tachyon/Proteus, redact identifiable customer data and
  state which version. We are an open project; we do not republish
  vendor-confidential output.
- **English-first, but multilingual is welcome.** Threads in other
  languages are fine. If a question gets traction, post a follow-up
  summary in English so the wider community benefits.
- **Cite when you can.** Linking the paper or repo a technique comes from
  is more useful than describing it from memory.
- **No proprietary data leaks.** This is the rule that gets someone
  removed fastest. If in doubt, do not paste it.

### Moderators

A small (3–5 person) moderator team enforces the etiquette rules above.
Three strikes = removal; the first strike is a private DM, not a public
callout.

If you want to be a moderator, the bar is: (a) a public technical
contribution to OpenLithoHub or an adjacent open-source project, and (b)
demonstrated technical writing. Self-nominate via GitHub Issue once the
channel is live.

## Onboarding flow

When the server opens, every new member sees:

1. A pinned `#welcome` post linking to this page and the
   [getting-started guide](getting-started.md).
2. A 60-second self-introduction template:
   ```
   Name (or handle):
   Affiliation (optional):
   What you work on:
   What you're hoping to get from this community:
   ```
3. A "first task" suggestion: pick one of the [good-first-issue
   labels](https://github.com/OpenLithoHub/OpenLithoHub/labels/good-first-issue)
   or run the [Colab BYOM notebook](https://colab.research.google.com/github/OpenLithoHub/OpenLithoHub/blob/main/notebooks/colab_byom.ipynb) and
   share a result in `#showcase`.

We deliberately do not gate posting behind self-introduction — gates kill
lurker-to-contributor conversion. The intro is a norm, not a wall.

## What this community is not

- **Not a tech-support channel for commercial EDA tools.** Calibre,
  Tachyon, Proteus questions belong on the respective vendor forums.
- **Not a recruiting board.** Job posts are removed. Ask in DM if you
  must.

## Code of Conduct

The Discord server follows the etiquette rules above. Violations can be
reported to the moderator team via DM or to `conduct@openlithohub.com`.

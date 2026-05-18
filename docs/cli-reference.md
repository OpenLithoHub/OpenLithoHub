# CLI Reference

OpenLithoHub provides a command-line interface via the `openlithohub` command.

## Global Options

```
openlithohub [OPTIONS] COMMAND [SUBCOMMAND] [ARGS]...
```

| Option | Description |
|--------|-------------|
| `--version`, `-V` | Print the installed version and exit. |

The CLI is organized into three command groups: `eval`, `optimize`, and
`leaderboard`. Each group exposes one or more subcommands; the most common is
`run` for the workflow groups.

## Commands

### `eval run` — Evaluate a Model

Run benchmark evaluation on a registered model.

```bash
openlithohub eval run [OPTIONS]
```

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--model`, `-m` | TEXT | Model name to evaluate (required). | — |
| `--dataset`, `-d` | TEXT | Dataset name (`lithobench` or `lithosim`). | `lithobench` |
| `--data-root`, `-r` | PATH | Path to dataset root directory (required). | — |
| `--output`, `-o` | PATH | Path to save the evaluation report. | unset |
| `--format`, `-f` | TEXT | Output format: `table`, `json`, or `markdown`. | `table` |
| `--node`, `-n` | TEXT | Process node for evaluation context. | `45nm` |
| `--pixel-nm` | FLOAT | Pixel size in nanometers. | `1.0` |
| `--limit`, `-l` | INT | Max samples to evaluate (default: all). | unset |
| `--mrc / --no-mrc` | FLAG | Run MRC compliance check. | `--mrc` |
| `--min-width-nm` | FLOAT | Minimum feature width for MRC (nm). | `40.0` |
| `--min-spacing-nm` | FLOAT | Minimum spacing for MRC (nm). | `40.0` |
| `--submit / --no-submit` | FLAG | Auto-submit results to leaderboard. | `--no-submit` |
| `--topology` | TEXT | Mask topology: `manhattan` or `curvilinear`. | `manhattan` |
| `--paper-url` | TEXT | Paper URL (used when submitting). | unset |
| `--code-url` | TEXT | Code/repo URL (used when submitting). | unset |

**Example:**

```bash
openlithohub eval run \
  --model dummy-identity \
  --dataset lithobench \
  --data-root ./data/lithobench \
  --format table
```

---

### `optimize run` — Run Optimization Pipeline

End-to-end mask optimization with OASIS export.

```bash
openlithohub optimize run [OPTIONS]
```

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--input`, `-i` | PATH | Input design file (`.oas` / `.gds`). Required. | — |
| `--model`, `-m` | TEXT | Optimization model to use. Required. | — |
| `--output`, `-o` | PATH | Output optimized layout path. Required. | — |
| `--writer`, `-w` | TEXT | Target writer: `mbmw` or `vsb`. | `mbmw` |
| `--node`, `-n` | TEXT | Target process node. | `3nm-euv` |
| `--drc-check` | FLAG | Run DRC/MRC checks after optimization. | off |
| `--tile-size` | INT | Tile size for distributed processing (pixels). | `2048` |
| `--overlap` | INT | Tile overlap for seamless stitching (pixels). | `128` |
| `--pixel-nm` | FLOAT | Pixel size in nanometers. | `1.0` |

**Example:**

```bash
openlithohub optimize run \
  --input design.oas \
  --model rule-based-opc \
  --writer mbmw \
  --node 3nm-euv \
  --drc-check \
  --output optimized.oas
```

---

### `leaderboard` — Manage Leaderboard

Subcommands for viewing, submitting, and exporting leaderboard data.

#### `leaderboard view`

Display current leaderboard rankings.

```bash
openlithohub leaderboard view [OPTIONS]
```

| Option | Type | Description |
|--------|------|-------------|
| `--dataset`, `-d` | TEXT | Filter by dataset. |
| `--node`, `-n` | TEXT | Filter by process node. |
| `--format`, `-f` | TEXT | Output format: `table`, `json`, `markdown`. |
| `--limit`, `-l` | INT | Max entries to display. |

#### `leaderboard submit`

Submit a benchmark result.

```bash
openlithohub leaderboard submit [OPTIONS]
```

| Option | Type | Description |
|--------|------|-------------|
| `--file`, `-F` | PATH | JSON file with `BenchmarkResult` fields. |
| `--model`, `-m` | TEXT | Model name. |
| `--dataset`, `-d` | TEXT | Dataset name. |
| `--node`, `-n` | TEXT | Process node. |
| `--topology`, `-t` | TEXT | `manhattan` or `curvilinear`. |
| `--epe-mean` | FLOAT | Mean EPE in nm. |
| `--epe-max` | FLOAT | Max EPE in nm. |
| `--pvband` | FLOAT | PV band width in nm. |
| `--paper-url` | TEXT | Paper URL. |
| `--code-url` | TEXT | Code/repo URL. |

**Example (from file):**

```bash
openlithohub leaderboard submit --file results.json
```

**Example (inline):**

```bash
openlithohub leaderboard submit \
  --model my-opc \
  --dataset lithobench \
  --node 3nm-euv \
  --topology curvilinear \
  --epe-mean 1.23 \
  --epe-max 4.56 \
  --pvband 2.1
```

#### `leaderboard export`

Export leaderboard to a file.

```bash
openlithohub leaderboard export [OPTIONS]
```

| Option | Type | Description |
|--------|------|-------------|
| `--output`, `-o` | PATH | Output file path (required). |
| `--format`, `-f` | TEXT | Export format: `json` or `markdown`. |
| `--dataset`, `-d` | TEXT | Filter by dataset. |
| `--node`, `-n` | TEXT | Filter by process node. |

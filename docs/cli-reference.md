# CLI Reference

OpenLithoHub provides a command-line interface via the `openlithohub` command.

## Global Options

```
openlithohub [OPTIONS] COMMAND [ARGS]...
```

## Commands

### `eval` — Evaluate a Model

Run benchmark evaluation on a registered model.

```bash
openlithohub eval [OPTIONS]
```

| Option | Type | Description |
|--------|------|-------------|
| `--model`, `-m` | TEXT | Model name (from registry) |
| `--dataset`, `-d` | TEXT | Dataset name (`lithobench` or `lithosim`) |
| `--data-root` | PATH | Path to dataset directory |
| `--format`, `-f` | TEXT | Output format: `table`, `json`, `csv` |
| `--metrics` | TEXT | Comma-separated metric list |

**Example:**

```bash
openlithohub eval \
  --model dummy-identity \
  --dataset lithobench \
  --data-root ./data/lithobench \
  --format table
```

---

### `optimize` — Run Optimization Pipeline

End-to-end mask optimization with OASIS export.

```bash
openlithohub optimize [OPTIONS]
```

| Option | Type | Description |
|--------|------|-------------|
| `--input`, `-i` | PATH | Input design file (.oas, .gds) |
| `--model`, `-m` | TEXT | Model name for optimization |
| `--writer` | TEXT | Mask writer type (`mbmw`, `vsb`) |
| `--node` | TEXT | Process node (`3nm-euv`, `5nm`, `7nm`) |
| `--drc-check` | FLAG | Enable DRC checking |
| `--output`, `-o` | PATH | Output OASIS file path |

**Example:**

```bash
openlithohub optimize \
  --input design.oas \
  --model my-opc \
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
| `--dataset`, `-d` | TEXT | Filter by dataset |
| `--node`, `-n` | TEXT | Filter by process node |
| `--format`, `-f` | TEXT | Output format: `table`, `json`, `markdown` |
| `--limit`, `-l` | INT | Max entries to display |

#### `leaderboard submit`

Submit a benchmark result.

```bash
openlithohub leaderboard submit [OPTIONS]
```

| Option | Type | Description |
|--------|------|-------------|
| `--file`, `-F` | PATH | JSON file with BenchmarkResult fields |
| `--model`, `-m` | TEXT | Model name |
| `--dataset`, `-d` | TEXT | Dataset name |
| `--node`, `-n` | TEXT | Process node |
| `--topology`, `-t` | TEXT | `manhattan` or `curvilinear` |
| `--epe-mean` | FLOAT | Mean EPE in nm |
| `--epe-max` | FLOAT | Max EPE in nm |
| `--pvband` | FLOAT | PV band width in nm |
| `--paper-url` | TEXT | Paper URL |
| `--code-url` | TEXT | Code/repo URL |

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
| `--output`, `-o` | PATH | Output file path (required) |
| `--format`, `-f` | TEXT | Export format: `json` or `markdown` |
| `--dataset`, `-d` | TEXT | Filter by dataset |
| `--node`, `-n` | TEXT | Filter by process node |

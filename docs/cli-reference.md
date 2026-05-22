# CLI Reference

OpenLithoHub provides a command-line interface via the `openlithohub` command.

## Global Options

```
openlithohub [OPTIONS] COMMAND [SUBCOMMAND] [ARGS]...
```

| Option | Description |
|--------|-------------|
| `--version`, `-V` | Print the installed version and exit. |

The CLI is organized into eight command groups: `eval`, `optimize`,
`leaderboard`, `simulate`, `synth`, `hackathon`, `export`, and `serve`.
Each group exposes one or more subcommands; the most common is `run`
for the workflow groups.

## Commands

### `eval run` — Evaluate a Model

Run benchmark evaluation on a registered model.

```bash
openlithohub eval run [OPTIONS]
```

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--model`, `-m` | TEXT | Model name to evaluate (required). | — |
| `--dataset`, `-d` | TEXT | Dataset name (`lithobench`, `lithosim`, `asap7`, `freepdk45`, or `orfs`). | `lithobench` |
| `--data-root`, `-r` | PATH | Path to dataset root directory (required). | — |
| `--output`, `-o` | PATH | Path to save the evaluation report. | unset |
| `--format`, `-f` | TEXT | Output format: `table`, `json`, or `markdown`. | `table` |
| `--node`, `-n` | TEXT | Process node for evaluation context. | `45nm` |
| `--pixel-nm` | FLOAT | Pixel size in nanometers. | `1.0` |
| `--limit`, `-l` | INT | Max samples to evaluate (default: all). | unset |
| `--drc / --no-drc` | FLAG | Run DRC compliance check. | `--drc` |
| `--mrc / --no-mrc` | FLAG | Run MRC compliance check. | `--mrc` |
| `--pvband / --no-pvband` | FLAG | Compute Process Variation Band metrics. | `--pvband` |
| `--min-width-nm` | FLOAT | Minimum feature width for MRC (nm). | `40.0` |
| `--min-spacing-nm` | FLOAT | Minimum spacing for MRC (nm). | `40.0` |
| `--accept-license` | FLAG | Acknowledge upstream PDK license — required for `asap7`, `freepdk45`, `orfs`. | off |
| `--tile-nm` | FLOAT | Tile edge length in nm (used by `--dataset orfs`; `2000` and `5000` are the canonical AI-OPC windows). | `2000.0` |
| `--device` | TEXT | Torch device for the forward model (`cpu`, `cuda`, `mps`). | `cpu` |
| `--dtype` | TEXT | Compute dtype for the forward model (`fp32`, `bf16`). | `fp32` |
| `--compile / --no-compile` | FLAG | Wrap the Hopkins forward with `torch.compile`. | `--compile` |
| `--pretrained / --no-pretrained` | FLAG | Load pretrained weights for the selected model (when supported). | `--no-pretrained` |
| `--sha256` | TEXT | Expected SHA256 digest for direct-URL weight downloads. | unset |
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
| `--halo` | TEXT | Tile halo: `auto` (default) computes `max(OIR_px, model_RF_px)` from `--node` and `--model`, or pass an integer. Mutually exclusive with `--overlap`. See [Halo and Tiling](halo-and-tiling.md). | `auto` |
| `--overlap` | INT | **Legacy.** Tile overlap in pixels. Kept for back-compat with pre-RFC-0005 scripts. Prefer `--halo`. | none |
| `--pixel-nm` | FLOAT | Pixel size in nanometers. | `1.0` |
| `--num-gpus` | INT | Worker processes for tile inference. `1` = sequential (default). `>1` spawns one worker per GPU and shards tiles round-robin. | `1` |

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

**Multi-GPU tile inference:**

`--num-gpus N` (`N>1`) shards tiles round-robin across `N` worker processes
spawned via `torch.multiprocessing` (spawn context, not fork — required for
CUDA safety). Each worker pins itself to one CUDA device (`cuda:rank`) when
enough GPUs are visible, and falls back to CPU dispatch otherwise so the
flag is testable without GPUs. Tile geometry and stitching stay in the
parent process; the model layer is unchanged, so ONNX/TorchScript export is
unaffected. Single-node only — multi-node scheduling is out of scope for
v0.3 (see RFC 0004).

```bash
openlithohub optimize run -i chip.oas -m neural-ilt --num-gpus 4 -o out.oas
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

---

### `simulate` — Forward Simulator

Run an optical forward simulator on a mask. The Hopkins backend ships in
the core install; vendor adapters (Calibre nmOPC, Tachyon) are
config-validated stubs that activate when the corresponding toolchain is
on `PATH`.

#### `simulate run`

Forward-simulate a mask with the selected backend.

```bash
openlithohub simulate run [OPTIONS] MASK_PATH
```

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `MASK_PATH` (positional) | PATH | Path to mask `.npy` or grayscale image. Required. | — |
| `--backend`, `-b` | TEXT | Simulator backend. | `hopkins` |
| `--out`, `-o` | PATH | Where to write the aerial image. | `aerial.npy` |
| `--pixel-size-nm` | FLOAT | Pixel size in nm. | `1.0` |
| `--wavelength-nm` | FLOAT | Exposure wavelength in nm. | `193.0` |
| `--na` | FLOAT | Numerical aperture. | `1.35` |
| `--sigma` | FLOAT | Outer partial-coherence factor. | `0.7` |
| `--threshold` | FLOAT | Resist threshold. | `0.225` |
| `--dose` | FLOAT | Dose multiplier. | `1.0` |

#### `simulate list-backends`

Print registered simulator backends. With `--verbose` (`-v`), each line
also includes the backing class path so you can locate the source
without grepping `simulators/registry.py`.

```bash
openlithohub simulate list-backends
openlithohub simulate list-backends --verbose
```

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--verbose`, `-v` | FLAG | Also print the implementing `module.ClassName`. | off |

---

### `synth` — Synthetic PDK-Aware Layouts

Generate hermetic synthetic layouts that pass MRC by construction. Useful
for CI, Colab, and quick iteration without dataset downloads.

#### `synth run`

Generate `n` synthetic layouts and write them as `.npy`.

```bash
openlithohub synth run [OPTIONS]
```

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--out`, `-o` | PATH | Output directory. | `synth_out` |
| `--pdk` | TEXT | PDK preset: `asap7` or `freepdk45`. | `freepdk45` |
| `--pattern`, `-p` | `sram` \| `contact_array` \| `random_logic` | Pattern type. | `random_logic` |
| `--n`, `-n` | INT | Number of layouts. | `10` |
| `--size`, `-s` | INT | Edge length in pixels. | `256` |
| `--seed` | INT | PRNG seed. | `0` |

#### `synth list-pdks`

Print registered PDK presets and their key rules.

```bash
openlithohub synth list-pdks
```

---

### `hackathon` — Inspect Current Contract

Print the active hackathon contract — frozen test-set tag, sample count,
hard gates, and target metric. Sourced from `hackathon/2026q3.yaml`.

#### `hackathon info`

```bash
openlithohub hackathon info [OPTIONS]
```

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--manifest`, `-m` | PATH | Path to manifest YAML. | repo default |

See the [Hackathon contributor guide](hackathon.md) for the full
submission flow.

---

### `export` — Export a Model

Export a registered model to a production-friendly artifact. Uses the
PyTorch `dynamo` ONNX path with a TorchScript fallback for models that
are not `torch.export`-able (e.g. Neural-ILT in some configurations).

#### `export run`

```bash
openlithohub export run [OPTIONS]
```

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--model`, `-m` | TEXT | Model name from the registry. Required. | — |
| `--format`, `-f` | TEXT | `onnx`, `torchscript`, or `tensorrt`. | `onnx` |
| `--output`, `-o` | PATH | Output file path. Required. | — |
| `--shape` | TEXT | Input HxW shape used to trace the model (e.g. `256x256`). | `256x256` |
| `--opset` | INT | ONNX opset version (ignored for non-ONNX formats). | `17` |
| `--pretrained / --no-pretrained` | FLAG | Load pretrained weights when supported. | `--no-pretrained` |
| `--device` | TEXT | Torch device to trace on (`cpu`, `cuda`, `mps`). | `cpu` |
| `--dynamic-batch / --static-batch` | FLAG | Mark the batch dimension as dynamic in the exported graph. | `--dynamic-batch` |

---

### `serve` — Run the HTTP Engine

Start a FastAPI micro-service exposing the OpenLithoHub optimization
engine. Models are loaded lazily on first request and cached in-process,
so repeat calls against the same model skip weight loading. Designed to
be embedded in fab-side schedulers (Slurm / LSF) and called from C++,
Java, or Perl pipelines that cannot embed the Python interpreter.

Install the optional dependency group first:

```bash
pip install "openlithohub[server]"
```

```bash
openlithohub serve [OPTIONS]
```

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--host` | TEXT | Bind address. | `127.0.0.1` |
| `--port`, `-p` | INT | TCP port. | `8000` |
| `--workers`, `-w` | INT | Uvicorn worker count. | `1` |
| `--log-level` | TEXT | uvicorn log level. | `info` |
| `--reload` | FLAG | Auto-reload on code changes (dev only). | off |

#### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/v1/health` | Liveness probe. |
| `GET`  | `/v1/models` | List registered model names. |
| `POST` | `/v1/optimize` | Multipart upload of a layout file + model name; returns the optimized layout binary. |

`POST /v1/optimize` form fields: `layout` (file), `model`, `node`,
`pixel_nm`, `tile_size`, `writer` (`mbmw` or `vsb`), `layer`
(`LAYER:DTYPE`, required for multi-layer files), `pretrained`. The
response carries `X-OLH-Tiles`, `X-OLH-Halo-Px`, `X-OLH-Export-Format`,
and `X-OLH-Shape` headers describing the run.

##### Example

```bash
openlithohub serve --port 8000 &

curl -X POST http://localhost:8000/v1/optimize \
     -F "layout=@chip.oas" \
     -F "model=neural-ilt" \
     -F "writer=mbmw" \
     -o optimized.oas
```

##### Interactive API explorer

The server is FastAPI under the hood, so it ships with two zero-config
docs surfaces:

| URL | What it is |
|-----|------------|
| `http://localhost:8000/docs` | Swagger UI — interactive playground; upload a layout, fire `/v1/optimize`, inspect headers and response body in the browser. |
| `http://localhost:8000/redoc` | ReDoc — read-only, three-pane reference rendering of the same OpenAPI schema. |
| `http://localhost:8000/openapi.json` | Raw OpenAPI 3.x schema — pipe into client generators (`openapi-generator`, `oapi-codegen`, etc.) to scaffold typed clients in any language. |

##### C++ client example (`cpp-httplib`)

Fab-side schedulers and OPC pipelines are usually C++. A self-contained
client with the header-only [`cpp-httplib`](https://github.com/yhirose/cpp-httplib)
is ~30 lines and links no extra runtime:

```cpp
// g++ -std=c++17 olh_client.cpp -o olh_client
#include <fstream>
#include <iostream>
#include <sstream>
#include "httplib.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: olh_client <input.oas> <output.oas>\n";
        return 1;
    }
    std::ifstream in(argv[1], std::ios::binary);
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string layout = buf.str();

    httplib::Client cli("http://localhost:8000");
    cli.set_read_timeout(600, 0);  // OPC tiles take time

    httplib::MultipartFormDataItems items = {
        {"layout", layout, argv[1], "application/octet-stream"},
        {"model",  "neural-ilt", "", ""},
        {"node",   "3nm-euv",    "", ""},
        {"writer", "mbmw",       "", ""},
    };

    auto res = cli.Post("/v1/optimize", items);
    if (!res || res->status != 200) {
        std::cerr << "optimize failed: "
                  << (res ? std::to_string(res->status) : "no response") << '\n';
        return 2;
    }
    std::ofstream out(argv[2], std::ios::binary);
    out.write(res->body.data(), res->body.size());

    std::cout << "tiles="    << res->get_header_value("X-OLH-Tiles")
              << " halo_px=" << res->get_header_value("X-OLH-Halo-Px")
              << " shape="   << res->get_header_value("X-OLH-Shape")
              << " format="  << res->get_header_value("X-OLH-Export-Format")
              << '\n';
    return 0;
}
```

For a `libcurl` build, substitute the body with a `CURLOPT_MIMEPOST`
multipart form (`curl_mime_init` → `curl_mime_addpart` per field). The
endpoint contract is the same — read the four `X-OLH-*` response
headers to drive the downstream MDP / OPC step.

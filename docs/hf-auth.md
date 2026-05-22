# HuggingFace authentication for gated datasets

Some datasets shipped through OpenLithoHub adapters live on the
HuggingFace Hub behind an access gate. This page is the one-stop guide
for unblocking them. Today this applies to:

| Adapter | Dataset | Reason |
|---------|---------|--------|
| `LithoSim` (`src/openlithohub/data/lithosim.py`) | [`OpenLithoHub/LithoSim`](https://huggingface.co/datasets/OpenLithoHub/LithoSim) | Sub-28 nm industrial layouts; access gated by the maintainers. |

If your `load()` call fails with `HTTP 401` or a `GatedRepoError`, the
adapter is telling you the current Python process is not authenticated
to the Hub. Follow the steps below.

## 1. Request access on the Hub

Open the dataset's Hub page (the link in the table above) while signed
in to your HuggingFace account, and click **Request access**. Approval
is manual on the maintainer side and may take a few hours.

You will know it succeeded when the dataset page shows a download
button instead of an access-request form.

## 2. Authenticate the local process

Once approved, the running Python process needs to present an HF token
on every request. Pick one of:

- **Interactive (recommended for laptops):**
  ```bash
  pip install -U "huggingface_hub[cli]"
  huggingface-cli login
  ```
  This writes a token to `~/.cache/huggingface/token`, and `datasets`
  / `huggingface_hub` will read it automatically. See the upstream
  guide for token-scope options:
  [HuggingFace — User access tokens](https://huggingface.co/docs/hub/security-tokens).

- **Environment variable (recommended for CI):**
  ```bash
  export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxx"
  ```
  `datasets.load_dataset(...)` will pick this up without further
  configuration. Most CI providers let you store the token as a
  masked secret and inject it into the job environment.

## 3. Verify

Re-run the load that previously failed. The adapter's 401 path should
no longer fire:

```python
from openlithohub.data.lithosim import LithoSim

ds = LithoSim().load()  # no RuntimeError
```

If it still fails, double-check that the account whose token you
exported is the same account that was granted access in step 1.

## Behind a corporate proxy

The token-exchange traffic with `huggingface.co` follows the same
`HTTPS_PROXY` / `HTTP_PROXY` rules as any other Hub call — see
[Networking & Proxies](networking.md) for the decision tree. Internal
proxy hostnames are intentionally **not** documented in the repository;
keep them in your shell or local config only.

## When the gate gets removed

If the maintainers ever flip `OpenLithoHub/LithoSim` back to public,
the adapter will silently start working without auth. This page stays
relevant for any future gated datasets — the steps are the same.

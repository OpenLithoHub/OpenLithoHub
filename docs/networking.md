# Networking and Proxies

OpenLithoHub fetches three classes of bytes from the public internet:

| Resource | Host(s) | Used by |
|----------|---------|---------|
| Source code & releases | `github.com` | `pip install`, `git clone` |
| Datasets, models | `huggingface.co` | `LithoSimDataset`, `huggingface-cli login` |
| Python packages | `pypi.org`, `files.pythonhosted.org` | `pip install` |

If you sit behind a corporate proxy or in a region with restricted access to
GitHub or HuggingFace, you will need to point the relevant clients at a proxy
that *can* reach those hosts. This page collects the configuration knobs in one
place so you don't have to discover them piecemeal.

## TL;DR

```bash
# Replace HOST:PORT with your own proxy. Use HTTPS_PROXY for outbound HTTPS.
export HTTPS_PROXY=http://HOST:PORT
export HTTP_PROXY=http://HOST:PORT
export NO_PROXY=localhost,127.0.0.1,.internal
```

These three variables are honoured by `git`, `pip`, the HuggingFace `datasets`
and `huggingface_hub` clients, `requests`, and most other Python tooling
OpenLithoHub depends on.

## Tool-by-tool configuration

### git

```bash
# Per-invocation (preferred — no global state):
HTTPS_PROXY=http://HOST:PORT git clone https://github.com/OpenLithoHub/OpenLithoHub.git

# Persistent (writes to ~/.gitconfig):
git config --global http.https://github.com.proxy http://HOST:PORT
```

Scoping the proxy to `https://github.com.` (note the trailing dot — it's
deliberate) avoids forcing internal git remotes through the same proxy.

### pip

```bash
pip install --proxy http://HOST:PORT openlithohub[data]
```

Or in `~/.pip/pip.conf`:

```ini
[global]
proxy = http://HOST:PORT
```

PyPI is often *faster* without the proxy than with it — only set this if pip is
actually unable to reach `pypi.org` directly. The HuggingFace and GitHub paths
below are usually the ones that need the proxy.

### HuggingFace (`datasets`, `huggingface_hub`)

```bash
export HTTPS_PROXY=http://HOST:PORT
huggingface-cli login        # one-time, for gated datasets like LithoSim
python -m openlithohub.cli download lithosim
```

The `datasets` library reads `HTTPS_PROXY` directly via its underlying
`requests` session — no library-specific knob needed.

For the LithoSim dataset specifically, see
[Getting Started → Datasets](getting-started.md) for the access-request flow.
The adapter detects 401/403 responses and emits a remediation hint pointing at
`huggingface-cli login`; if you see that message and you *are* logged in, the
proxy is the next thing to check.

### Docker / container builds

```dockerfile
ARG HTTPS_PROXY
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV HTTP_PROXY=${HTTPS_PROXY}
RUN pip install openlithohub[data]
```

Build with: `docker build --build-arg HTTPS_PROXY=http://HOST:PORT .`

## Authenticated proxies

If your proxy requires a username/password:

```bash
export HTTPS_PROXY=http://USER:PASSWORD@HOST:PORT
```

Be mindful that this puts credentials in your shell history and process
listings. Prefer a [pac-file]-aware client (e.g. `cntlm`, `px`) that holds the
credentials in a separate config file and exposes a local unauthenticated
endpoint for `HTTPS_PROXY` to point at.

[pac-file]: https://en.wikipedia.org/wiki/Proxy_auto-config

## Verifying connectivity

```bash
# GitHub:
curl -sSI https://github.com/OpenLithoHub/OpenLithoHub | head -1

# HuggingFace:
curl -sSI https://huggingface.co/datasets/OpenLithoHub/LithoSim | head -1

# PyPI:
curl -sSI https://pypi.org/simple/openlithohub/ | head -1
```

A successful response is `HTTP/2 200` (or `301`/`302`); a hung connection or
`Could not resolve host` means the proxy isn't reaching the target.

## What we don't ship

OpenLithoHub does not bundle a proxy host or auto-detect one — there's no
sensible default that works across networks. Set `HTTPS_PROXY` yourself when
your environment requires it, and unset it when it doesn't.

# Data Adapters

!!! info "Authentication for gated datasets"
    Some adapters (currently `LithoSim`) load data from a gated
    HuggingFace Hub repository. If a `load()` call raises a
    `RuntimeError` mentioning HTTP 401, see
    **[HuggingFace Authentication](../hf-auth.md)** for the unblock
    steps (request access → `huggingface-cli login` / `HF_TOKEN`).

::: openlithohub.data.base
    options:
      show_root_heading: true
      members_order: source

::: openlithohub.data.lithobench
    options:
      show_root_heading: true
      members_order: source

::: openlithohub.data.lithosim
    options:
      show_root_heading: true
      members_order: source

::: openlithohub.data.iccad16
    options:
      show_root_heading: true
      members_order: source

::: openlithohub.data.ganopc
    options:
      show_root_heading: true
      members_order: source

::: openlithohub.data.asap7
    options:
      show_root_heading: true
      members_order: source

::: openlithohub.data.freepdk45
    options:
      show_root_heading: true
      members_order: source

::: openlithohub.data.orfs
    options:
      show_root_heading: true
      members_order: source

::: openlithohub.data.transforms
    options:
      show_root_heading: true
      members_order: source

::: openlithohub.data.dummy
    options:
      show_root_heading: true
      members_order: source

# Model Interface

::: openlithohub.models.base
    options:
      show_root_heading: true
      members_order: source

::: openlithohub.models.registry
    options:
      show_root_heading: true
      members_order: source

::: openlithohub.models.levelset_ilt
    options:
      show_root_heading: true
      members_order: source

## Differentiable forward models

::: openlithohub._utils.forward_model
    options:
      show_root_heading: true
      members_order: source
      members:
        - simulate_aerial_image
        - apply_resist_threshold

::: openlithohub._utils.hopkins
    options:
      show_root_heading: true
      members_order: source
      members:
        - HopkinsParams
        - compute_socs_kernels
        - simulate_aerial_image_hopkins
        - clear_kernel_cache

::: openlithohub._utils.resist_model
    options:
      show_root_heading: true
      members_order: source
      members:
        - differentiable_threshold
        - simulate_resist
        - simulate_resist_soft

# Simulator backends

`openlithohub.simulators` is the vendor-neutral interface for lithography
forward simulation. The bundled `HopkinsSimulator` is the reference
implementation; `CalibreSimulator` and `TachyonSimulator` are commercial
adapters that require the respective vendor toolchain on `PATH` and a
license. Both support `mock_mode` (`config.extra["mock_mode"] = True`)
for testing and CI without the commercial binary.

The module also re-exports `load_source_intensity`,
`load_zernike_coefficients`, and `zernike_phase_map` so callers can drive
the Hopkins forward model with measured-source maps and Zernike-pupil
aberrations without reaching into `_utils.optics`.

## Registry

::: openlithohub.simulators.registry
    options:
      show_root_heading: true
      members_order: source
      members:
        - get_simulator
        - list_simulators
        - register_simulator
        - describe_simulators

## Base interface

::: openlithohub.simulators.base
    options:
      show_root_heading: true
      members_order: source
      members:
        - BaseSimulator
        - SimulatorConfig
        - SimulatorResult

## Hopkins reference adapter

::: openlithohub.simulators.hopkins_sim
    options:
      show_root_heading: true
      members_order: source
      members:
        - HopkinsSimulator

## Commercial adapters

::: openlithohub.simulators.commercial
    options:
      show_root_heading: true
      members_order: source
      members:
        - CommercialSimulatorAdapter
        - PreflightStatus
        - ToolchainError

::: openlithohub.simulators.calibre
    options:
      show_root_heading: true
      members_order: source
      members:
        - CalibreSimulator

::: openlithohub.simulators.tachyon
    options:
      show_root_heading: true
      members_order: source
      members:
        - TachyonSimulator

## Measured-source / Zernike-pupil I/O

::: openlithohub._utils.optics
    options:
      show_root_heading: true
      members_order: source
      members:
        - load_source_intensity
        - load_zernike_coefficients
        - zernike_phase_map

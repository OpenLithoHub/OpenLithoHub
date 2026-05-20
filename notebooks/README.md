# OpenLithoHub Notebooks

Hands-on tutorials. Click the badges to open them in Google Colab — no local
install required.

| Notebook | Description | Colab |
| --- | --- | --- |
| [`quickstart.ipynb`](quickstart.ipynb) | Install → dummy layout → metrics → paper figure in under 3 minutes. Avoids KLayout / scipy so it runs on Colab's stock runtime. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OpenLithoHub/OpenLithoHub/blob/main/notebooks/quickstart.ipynb) |
| [`colab_byom.ipynb`](colab_byom.ipynb) | Bring-your-own-model walkthrough: subclass `LithographyModel`, register it, and run the full benchmark suite from a fresh Colab. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OpenLithoHub/OpenLithoHub/blob/main/notebooks/colab_byom.ipynb) |
| [`auto_calibration.ipynb`](auto_calibration.ipynb) | Inverse-fit resist threshold + Gaussian σ to a synthetic gauge table using `torch.optim.Adam`. CPU, <30 s, pre-fit MAE 1.999 px → ~0 px after fit. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OpenLithoHub/OpenLithoHub/blob/main/notebooks/auto_calibration.ipynb) |

These notebooks pin only the `[jupyter]` extra. Workflows that depend on the
KLayout-backed exporter or the scipy-backed B-spline fitter are demonstrated in
the main docs at <https://docs.openlithohub.com>.

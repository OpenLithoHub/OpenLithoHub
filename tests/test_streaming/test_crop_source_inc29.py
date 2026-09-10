from dataclasses import dataclass

import torch

from openlithohub.streaming.crop_source import ExactVectorCropSource
from openlithohub.streaming.geometry import BoundingBox
from openlithohub.verify.interface_runs import HorizontalRun


@dataclass
class Parent:
    shape: tuple[int, int] = (20, 30)
    pixel_size_nm: float = 2.0
    layout_hash: str = "parent-hash"
    backend_kind: str = "fake-exact"

    def iter_runs_for_bbox(self, bbox_px):
        x0, y0, x1, y1 = bbox_px
        runs = (
            HorizontalRun(5, 3, 12),
            HorizontalRun(8, 9, 22),
            HorizontalRun(14, 1, 28),
        )
        out = []
        for r in runs:
            if y0 <= r.y < y1:
                a = max(x0, r.x0)
                b = min(x1, r.x1)
                if b > a:
                    out.append(HorizontalRun(r.y, a, b))
        return tuple(out)

    def read_window(self, bbox):
        arr = torch.zeros((bbox.height, bbox.width))
        for r in self.iter_runs_for_bbox((bbox.x0, bbox.y0, bbox.x1, bbox.y1)):
            arr[r.y - bbox.y0, r.x0 - bbox.x0 : r.x1 - bbox.x0] = 1
        return arr


def test_crop_translates_run_coordinates():
    crop = ExactVectorCropSource(
        Parent(),
        BoundingBox(7, 4, 23, 16),
    )
    assert crop.shape == (12, 16)
    runs = tuple(crop.iter_runs_for_bbox((0, 0, 16, 12)))
    assert runs == (
        HorizontalRun(1, 0, 5),
        HorizontalRun(4, 2, 15),
        HorizontalRun(10, 0, 16),
    )


def test_crop_read_window_matches_parent_window():
    parent = Parent()
    crop = ExactVectorCropSource(
        parent,
        BoundingBox(7, 4, 23, 16),
    )
    local = BoundingBox(2, 1, 10, 7)
    got = crop.read_window(local)
    want = parent.read_window(BoundingBox(9, 5, 17, 11))
    assert torch.equal(got, want)


def test_crop_subwindow_run_query_is_clipped_locally():
    crop = ExactVectorCropSource(
        Parent(),
        BoundingBox(7, 4, 23, 16),
    )
    runs = tuple(crop.iter_runs_for_bbox((4, 0, 9, 6)))
    assert runs == (
        HorizontalRun(1, 4, 5),
        HorizontalRun(4, 4, 9),
    )


def test_crop_rejects_parent_escape():
    try:
        ExactVectorCropSource(
            Parent(),
            BoundingBox(-1, 0, 10, 10),
        )
    except ValueError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("expected fail-closed crop bounds")

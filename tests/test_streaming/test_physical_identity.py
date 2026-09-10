from openlithohub.streaming.physical_identity import (
    InstancePathElement,
    PhysicalInstanceKey,
    ReadViewKey,
)


def key(member_i: int) -> PhysicalInstanceKey:
    return PhysicalInstanceKey(
        source_format="gds",
        source_cell_index=7,
        source_shape_fingerprint="shape-a",
        instance_path=(
            InstancePathElement(
                source_cell_index=11,
                array_i=member_i,
                array_j=0,
                specific_transform=f"r0 1000,{member_i * 100}",
            ),
        ),
        quantization_policy="exact-integer-dbu-per-pixel:8",
    )


def test_distinct_source_array_members_are_distinct_physical_owners():
    assert key(0).owner_id != key(1).owner_id


def test_same_physical_instance_read_in_two_tiles_keeps_owner():
    owner = key(0).owner_id
    a = ReadViewKey(owner, "tile-a", (0, 0, 16, 16))
    b = ReadViewKey(owner, "tile-b", (8, 0, 24, 16))
    assert a.physical_owner_id == b.physical_owner_id
    assert a.token != b.token


def test_source_format_is_part_of_physical_provenance():
    gds = key(0)
    oas = PhysicalInstanceKey(
        source_format="oas",
        source_cell_index=gds.source_cell_index,
        source_shape_fingerprint=gds.source_shape_fingerprint,
        instance_path=gds.instance_path,
        quantization_policy=gds.quantization_policy,
    )
    assert gds.owner_id != oas.owner_id

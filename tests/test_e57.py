"""Cross-check the pure-numpy reader against pye57 (libE57Format)."""

import glob
import os

import e57
import numpy as np
import pye57
import pytest
from e57 import E57Error, E57File, _unpack_bits

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
EXTRA = os.environ.get("E57_EXTRA_DIR", os.path.expanduser("~/Downloads"))

VALID = [
    "bunnyDouble.e57",
    "bunnyInt32.e57",
    "ColouredCubeDouble.e57",
    "ColouredCubeFloat.e57",
    "ColourRepresentation.e57",
    "ZeroPoints.e57",
]
INVALID = ["bad-crc.e57", "NoPrototype.e57"]

valid_paths = [os.path.join(DATA, f) for f in VALID] + sorted(glob.glob(os.path.join(EXTRA, "*.e57")))


@pytest.fixture(autouse=True, scope="session")
def _check_data():
    if not os.path.exists(os.path.join(DATA, "bunnyInt32.e57")):
        pytest.exit("Run tests/fetch_data.sh first", returncode=1)


@pytest.mark.parametrize("path", valid_paths, ids=os.path.basename)
def test_matches_pye57(path):
    ref = pye57.E57(path)
    with E57File(path) as f:
        assert len(f.scans) == ref.scan_count
        for i, scan in enumerate(f.scans):
            header = ref.get_header(i)
            assert scan.record_count == header.point_count
            ours = f.read_points(scan)
            for name in scan.field_names:
                assert len(ours[name]) == scan.record_count
            if not scan.record_count:
                continue
            # pye57 skips extension fields and stores colour as uint8, so compare what it can hold.
            names = [n for n in scan.field_names if n in pye57.e57.SUPPORTED_POINT_FIELDS]
            if any(ours[n].max() > 255 for n in names if n.startswith("color")):
                names = [n for n in names if not n.startswith("color")]
            theirs, buffers = ref.make_buffers(names, scan.record_count)
            header.points.reader(buffers).read()
            for name in names:
                np.testing.assert_allclose(ours[name], theirs[name], rtol=0, atol=1e-9, err_msg=name)
            np.testing.assert_allclose(scan.rotation_matrix(), header.rotation_matrix, atol=1e-12)
            np.testing.assert_allclose(scan.translation, header.translation, atol=1e-12)


def test_extension_field_and_odd_bit_widths():
    with E57File(os.path.join(DATA, "ColourRepresentation.e57")) as f:
        scan = f.scans[0]
        widths = {s.name: s.bits for s in scan.fields}
        assert widths["returnIndex"] == 1 and widths["cartesianX"] == 10
        pts = f.read_points(scan, ["las:pointSourceId", "colorRed"])
        assert set(pts) == {"las:pointSourceId", "colorRed"}
        assert pts["colorRed"].max() > 255  # 16-bit colour


@pytest.mark.parametrize("name", INVALID)
def test_invalid_files_raise(name):
    with pytest.raises(E57Error), E57File(os.path.join(DATA, name)) as f:
        for s in f.scans:
            f.read_points(s)


@pytest.mark.parametrize("bits", [1, 3, 7, 10, 13, 24, 31, 33, 47, 56, 57, 63])
def test_unpack_bits_roundtrip(bits):
    rng = np.random.default_rng(bits)
    n = 1001
    vals = rng.integers(0, 2**bits, size=n, dtype=np.uint64, endpoint=False) if bits < 64 else None
    # Pack LSB-first with Python ints as an independent reference.
    acc = 0
    for i, v in enumerate(vals.tolist()):
        acc |= v << (i * bits)
    data = acc.to_bytes((n * bits + 7) // 8, "little")
    np.testing.assert_array_equal(_unpack_bits(data, bits, n), vals)


@pytest.mark.parametrize("name", ["empty.e57", "ZeroPoints.e57", "InvalidCVHeader.e57"])
def test_files_without_points(name):
    with E57File(os.path.join(DATA, name)) as f:
        for s in f.scans:
            assert all(len(v) == 0 for v in f.read_points(s).values())


def test_spherical_to_cartesian():
    x, y, z = e57.spherical_to_cartesian(np.array([2.0]), np.array([np.pi / 2]), np.array([0.0]))
    np.testing.assert_allclose([x[0], y[0], z[0]], [0, 2, 0], atol=1e-12)


def test_multi_scan_spherical_with_pose(tmp_path):
    path = str(tmp_path / "multi.e57")
    rng = np.random.default_rng(0)
    scans = []
    w = pye57.E57(path, mode="w")
    for k in range(2):
        n = 500 + k
        data = {
            "sphericalRange": rng.uniform(1, 50, n),
            "sphericalAzimuth": rng.uniform(-np.pi, np.pi, n),
            "sphericalElevation": rng.uniform(-1.2, 1.2, n),
            "intensity": rng.uniform(0, 1, n).astype(np.float32),
            "rowIndex": rng.integers(0, 1000, n).astype(np.uint16),
            "columnIndex": rng.integers(0, 70, n).astype(np.uint16),
        }
        x, y, z = e57.spherical_to_cartesian(
            data["sphericalRange"], data["sphericalAzimuth"], data["sphericalElevation"]
        )
        data.update(cartesianX=x, cartesianY=y, cartesianZ=z)
        q = np.array([np.cos(0.3 * (k + 1)), 0.0, 0.0, np.sin(0.3 * (k + 1))])
        t = np.array([10.0 * k, -5.0, 1.5])
        w.write_scan_raw(data, name=f"s{k}", rotation=q, translation=t)
        scans.append((data, q, t))
    w.close()

    with E57File(path) as f:
        assert [s.name for s in f.scans] == ["s0", "s1"]
        for scan, (data, q, t) in zip(f.scans, scans):
            pts = f.read_points(scan)
            assert {"cartesianX", "intensity", "rowIndex"} <= set(pts)
            for name in pts:  # pye57's writer drops the spherical fields
                np.testing.assert_allclose(pts[name], data[name], rtol=1e-6, err_msg=name)
            np.testing.assert_allclose(scan.rotation, q)
            np.testing.assert_allclose(scan.translation, t)

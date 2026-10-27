"""Pure-Python (numpy) reader for ASTM E57 point cloud files.

No bpy imports here, so the reader can be tested outside Blender.

File layout (ASTM E2807):
  * The file is split into physical pages (normally 1024 bytes); the last 4 bytes
    of each page are a CRC32C checksum, the rest is "logical" payload.
  * A 48-byte header points at an XML section describing the data tree.
  * Point records live in binary CompressedVector sections, as data packets
    holding one bytestream per prototype field. Each bytestream is a
    little-endian, LSB-first bit-packed stream that continues across packets.
"""

from __future__ import annotations

import io
import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

E57_NS = "http://www.astm.org/COMMIT/E57/2010-e57-v1.0"

_HEADER = struct.Struct("<8sIIQQQQ")
_SECTION_HEADER = struct.Struct("<B7xQQQ")
_DATA_PACKET = 1
_INDEX_PACKET = 0
_EMPTY_PACKET = 2

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class E57Error(Exception):
    pass


def _crc32c_table() -> list[int]:
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
        table.append(c)
    return table


_CRC32C_TABLE = _crc32c_table()


def crc32c(data: bytes) -> int:
    # Pure Python, so only used on the few pages holding the header and XML.
    c = 0xFFFFFFFF
    table = _CRC32C_TABLE
    for b in data:
        c = table[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


# --------------------------------------------------------------------------- XML tree


@dataclass
class Node:
    """One element of the E57 XML tree."""

    name: str
    type: str
    attrs: dict
    text: str
    children: list[Node]

    def child(self, name: str) -> Node | None:
        for c in self.children:
            if c.name == name:
                return c
        return None

    def value(self, name: str, default=None):
        """Typed value of a direct child leaf (Integer/Float/ScaledInteger/String)."""
        c = self.child(name)
        return default if c is None else c.leaf_value()

    def leaf_value(self):
        text = self.text.strip()
        if self.type == "Integer":
            return int(text) if text else 0
        if self.type == "Float":
            return float(text) if text else 0.0
        if self.type == "ScaledInteger":
            raw = int(text) if text else 0
            return raw * float(self.attrs.get("scale", 1.0)) + float(self.attrs.get("offset", 0.0))
        return self.text


def _parse_xml(xml_bytes: bytes) -> Node:
    # Keep "prefix:local" names for extension namespaces (e.g. "nor:normalX"),
    # and bare local names for the default E57 namespace.
    prefixes: dict[str, str] = {}
    root_elem = None

    for event, item in ET.iterparse(io.BytesIO(xml_bytes), events=("start-ns", "start")):
        if event == "start-ns":
            prefix, uri = item
            prefixes.setdefault(uri, prefix)
        elif root_elem is None:
            root_elem = item

    def local(tag: str) -> str:
        if tag.startswith("{"):
            uri, name = tag[1:].split("}", 1)
            if uri == E57_NS or not prefixes.get(uri):
                return name
            return f"{prefixes[uri]}:{name}"
        return tag

    def convert(elem) -> Node:
        attrs = {local(k): v for k, v in elem.attrib.items()}
        return Node(
            name=local(elem.tag),
            type=attrs.get("type", ""),
            attrs=attrs,
            text=elem.text or "",
            children=[convert(c) for c in elem],
        )

    if root_elem is None:
        raise E57Error("Empty XML section")
    return convert(root_elem)


# --------------------------------------------------------------------------- fields


@dataclass
class FieldSpec:
    """One prototype field of a CompressedVector (one bytestream per record)."""

    name: str
    type: str  # "Float" | "Integer" | "ScaledInteger"
    bits: int = 0  # bit width for Integer / ScaledInteger
    double: bool = True  # Float precision
    minimum: int = 0
    maximum: int = 0
    scale: float = 1.0
    offset: float = 0.0

    @property
    def bytestream_bits(self) -> int:
        if self.type == "Float":
            return 64 if self.double else 32
        return self.bits

    @classmethod
    def from_node(cls, node: Node) -> FieldSpec:
        a = node.attrs
        if node.type == "Float":
            return cls(node.name, "Float", double=a.get("precision", "double") != "single")
        if node.type in ("Integer", "ScaledInteger"):
            lo = int(a.get("minimum", _INT64_MIN))
            hi = int(a.get("maximum", _INT64_MAX))
            if hi < lo:
                raise E57Error(f"Field {node.name}: maximum < minimum")
            return cls(
                node.name,
                node.type,
                bits=(hi - lo).bit_length(),
                minimum=lo,
                maximum=hi,
                scale=float(a.get("scale", 1.0)),
                offset=float(a.get("offset", 0.0)),
            )
        raise E57Error(f"Unsupported prototype field type {node.type!r} for {node.name}")


@dataclass
class Scan:
    """A data3D entry: one point cloud with its pose and metadata."""

    index: int
    name: str
    guid: str
    record_count: int
    file_offset: int
    fields: list[FieldSpec]
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # w, x, y, z
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    color_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    intensity_limits: tuple[float, float] | None = None
    node: Node | None = None

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    def has(self, *names: str) -> bool:
        return all(n in self.field_names for n in names)

    def rotation_matrix(self) -> np.ndarray:
        w, x, y, z = self.rotation
        n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
        w, x, y, z = w / n, x / n, y / n, z / n
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )


# --------------------------------------------------------------------------- file


class E57File:
    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")
        try:
            self._read_header()
            self.root = _parse_xml(
                self.read_logical_at_physical(self.xml_offset, self.xml_length, verify=True)
            )
            self.scans = self._collect_scans()
        except Exception:
            self._f.close()
            raise

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ paging

    def _read_header(self):
        raw = self._f.read(_HEADER.size)
        if len(raw) < _HEADER.size:
            raise E57Error("File too small to be E57")
        sig, major, minor, phys_len, xml_off, xml_len, page_size = _HEADER.unpack(raw)
        if sig != b"ASTM-E57":
            raise E57Error("Not an E57 file (bad signature)")
        if major != 1:
            raise E57Error(f"Unsupported E57 version {major}.{minor}")
        if page_size <= 4:
            raise E57Error(f"Invalid page size {page_size}")
        self.page_size = page_size
        self.payload_size = page_size - 4
        self.read_logical(0, 1, verify=True)  # header page
        self._f.seek(0, 2)
        if self._f.tell() < phys_len:
            raise E57Error("File is truncated")
        self.version = (major, minor)
        self.xml_offset = xml_off
        self.xml_length = xml_len

    def _logical(self, physical: int) -> int:
        return (physical // self.page_size) * self.payload_size + physical % self.page_size

    def read_logical(self, logical: int, n: int, verify: bool = False) -> bytes:
        """Read n payload bytes starting at a logical offset, skipping page CRCs.

        With verify, each page's CRC32C (stored big-endian in its last 4 bytes) is checked.
        """
        if n <= 0:
            return b""
        ps, pl = self.page_size, self.payload_size
        first = logical // pl
        last = (logical + n - 1) // pl
        self._f.seek(first * ps)
        raw = self._f.read((last - first + 1) * ps)
        if len(raw) % ps:
            raw += b"\0" * (ps - len(raw) % ps)
        pages = np.frombuffer(raw, np.uint8).reshape(-1, ps)
        if verify:
            for i, page in enumerate(pages):
                if crc32c(page[:pl].tobytes()) != int.from_bytes(page[pl:].tobytes(), "big"):
                    raise E57Error(f"Checksum mismatch on page {first + i}, file is corrupted")
        pages = pages[:, :pl]
        start = logical - first * pl
        out = pages.reshape(-1)[start : start + n].tobytes()
        if len(out) < n:
            raise E57Error("Unexpected end of file")
        return out

    def read_logical_at_physical(self, physical: int, n: int, verify: bool = False) -> bytes:
        return self.read_logical(self._logical(physical), n, verify)

    # ------------------------------------------------------------------ scans

    def _collect_scans(self) -> list[Scan]:
        data3d = self.root.child("data3D")
        scans = []
        for i, node in enumerate(data3d.children if data3d else []):
            points = node.child("points")
            if points is None or points.type != "CompressedVector":
                continue
            proto = points.child("prototype")
            if proto is None:
                raise E57Error(f"Scan {i}: missing prototype")
            scan = Scan(
                index=i,
                name=node.value("name") or f"scan_{i}",
                guid=node.value("guid") or "",
                record_count=int(points.attrs.get("recordCount", 0)),
                file_offset=int(points.attrs.get("fileOffset", 0)),
                fields=[FieldSpec.from_node(c) for c in proto.children],
                node=node,
            )
            pose = node.child("pose")
            if pose is not None:
                rot, tr = pose.child("rotation"), pose.child("translation")
                if rot is not None:
                    scan.rotation = tuple(rot.value(k, 0.0) for k in "wxyz")
                if tr is not None:
                    scan.translation = tuple(tr.value(k, 0.0) for k in "xyz")
            limits = node.child("colorLimits")
            if limits is not None:
                for c in ("Red", "Green", "Blue"):
                    lo = limits.value(f"color{c}Minimum")
                    hi = limits.value(f"color{c}Maximum")
                    if lo is not None and hi is not None and hi > lo:
                        scan.color_limits[f"color{c}"] = (float(lo), float(hi))
            limits = node.child("intensityLimits")
            if limits is not None:
                lo, hi = limits.value("intensityMinimum"), limits.value("intensityMaximum")
                if lo is not None and hi is not None and hi > lo:
                    scan.intensity_limits = (float(lo), float(hi))
            scans.append(scan)
        return scans

    # ------------------------------------------------------------------ points

    def read_points(self, scan: Scan, names=None, progress=None) -> dict[str, np.ndarray]:
        """Decode the requested prototype fields (all by default) of a scan.

        Returns {field name: 1D array of length scan.record_count}. Integers come back
        as int64, ScaledIntegers and Floats as float64 (Floats keep float32 when single).
        `progress(fraction)` is called while reading packets.
        """
        n = scan.record_count
        wanted = set(scan.field_names if names is None else names)
        missing = wanted - set(scan.field_names)
        if missing:
            raise E57Error(f"Scan {scan.name!r} has no fields {sorted(missing)}")
        if n == 0:
            return {f.name: _empty_for(f) for f in scan.fields if f.name in wanted}

        streams = self._read_bytestreams(scan, wanted, progress)
        out = {}
        for i, spec in enumerate(scan.fields):
            if spec.name in wanted:
                out[spec.name] = _decode_field(spec, b"".join(streams[i]), n)
        return out

    def _read_bytestreams(self, scan: Scan, wanted: set[str], progress) -> list[list[bytes]]:
        header = self.read_logical_at_physical(scan.file_offset, _SECTION_HEADER.size)
        section_id, section_len, data_phys, _index_phys = _SECTION_HEADER.unpack(header)
        if section_id != 1:
            raise E57Error(f"Scan {scan.name!r}: bad CompressedVector section header")
        section_end = self._logical(scan.file_offset) + section_len
        pos = self._logical(data_phys)

        nfields = len(scan.fields)
        needed = [
            (scan.record_count * f.bytestream_bits + 7) // 8 if f.name in wanted else 0 for f in scan.fields
        ]
        got = [0] * nfields
        streams: list[list[bytes]] = [[] for _ in range(nfields)]
        total = max(section_end - pos, 1)
        start = pos

        while pos + 4 <= section_end and any(g < w for g, w in zip(got, needed)):
            ptype, _flags, length_m1 = struct.unpack("<BBH", self.read_logical(pos, 4))
            length = length_m1 + 1
            if ptype == _DATA_PACKET:
                packet = self.read_logical(pos, length)
                (count,) = struct.unpack_from("<H", packet, 4)
                if count != nfields:
                    raise E57Error(
                        f"Scan {scan.name!r}: packet has {count} bytestreams, prototype has {nfields}"
                    )
                sizes = struct.unpack_from(f"<{count}H", packet, 6)
                off = 6 + 2 * count
                for i, size in enumerate(sizes):
                    if size and needed[i]:
                        streams[i].append(packet[off : off + size])
                        got[i] += size
                    off += size
            elif ptype not in (_INDEX_PACKET, _EMPTY_PACKET):
                raise E57Error(f"Scan {scan.name!r}: unknown packet type {ptype}")
            pos += length
            if progress is not None:
                progress((pos - start) / total)

        for i, f in enumerate(scan.fields):
            if got[i] < needed[i]:
                raise E57Error(
                    f"Scan {scan.name!r}: field {f.name} truncated ({got[i]} of {needed[i]} bytes)"
                )
        return streams


def _empty_for(spec: FieldSpec) -> np.ndarray:
    if spec.type == "Integer":
        return np.zeros(0, np.int64)
    if spec.type == "Float" and not spec.double:
        return np.zeros(0, np.float32)
    return np.zeros(0, np.float64)


def _unpack_bits(data: bytes, bits: int, n: int) -> np.ndarray:
    """Unpack n unsigned integers of `bits` width from an LSB-first bitstream."""
    if bits == 0:
        return np.zeros(n, np.uint64)
    if bits in (8, 16, 32, 64):
        return np.frombuffer(data, f"<u{bits // 8}", count=n).astype(np.uint64)
    buf = np.frombuffer(data, np.uint8)
    if bits > 56:
        # Rare: a value can span 9 bytes, so go through an explicit bit array.
        flat = np.unpackbits(buf, bitorder="little")[: n * bits].reshape(n, bits)
        weights = np.left_shift(np.uint64(1), np.arange(bits, dtype=np.uint64))
        return (flat.astype(np.uint64) * weights).sum(axis=1, dtype=np.uint64)
    buf = np.concatenate([buf, np.zeros(8, np.uint8)])
    bit_pos = np.arange(n, dtype=np.uint64) * np.uint64(bits)
    byte_pos = (bit_pos >> np.uint64(3)).astype(np.intp)
    shift = bit_pos & np.uint64(7)
    span = (bits + 7 + 7) // 8  # bytes touched by one value, including the bit shift
    acc = np.zeros(n, np.uint64)
    for k in range(span):
        acc |= buf[byte_pos + k].astype(np.uint64) << np.uint64(8 * k)
    return (acc >> shift) & np.uint64((1 << bits) - 1)


def _decode_field(spec: FieldSpec, data: bytes, n: int) -> np.ndarray:
    if spec.type == "Float":
        return np.frombuffer(data, "<f8" if spec.double else "<f4", count=n).copy()
    raw = _unpack_bits(data, spec.bits, n)
    if spec.type == "Integer":
        if spec.minimum >= 0 or spec.bits < 64:
            return raw.astype(np.int64) + spec.minimum
        return (raw + np.uint64(spec.minimum & (2**64 - 1))).view(np.int64)
    # ScaledInteger: compute in float64 to avoid int overflow on minimum offset.
    if spec.bits < 64:
        ints = raw.astype(np.int64) + spec.minimum
    else:
        ints = (raw + np.uint64(spec.minimum & (2**64 - 1))).view(np.int64)
    return ints.astype(np.float64) * spec.scale + spec.offset


# --------------------------------------------------------------------------- helpers


def spherical_to_cartesian(r, azimuth, elevation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cos_el = np.cos(elevation)
    return r * cos_el * np.cos(azimuth), r * cos_el * np.sin(azimuth), r * np.sin(elevation)

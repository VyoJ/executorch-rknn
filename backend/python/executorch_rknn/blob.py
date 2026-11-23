"""The delegate blob: a small header describing the delegate's inputs and
outputs as ExecuTorch sees them, followed by the .rknn model.

All integers little-endian.

  magic    4 bytes   b"RKNB"
  version  u16       1
  n_in     u16
  n_out    u16
  reserved u16
  per input, then per output:
    dtype  u8        see DTYPES
    ndim   u8
    pad    u16
    dims   ndim x u32
  rknn_size u64
  rknn      rknn_size bytes

The runtime queries everything else (layout, quantisation) from RKNN.
"""

import struct
from dataclasses import dataclass

MAGIC = b"RKNB"
VERSION = 1
DTYPES = {"float32": 0, "int8": 1, "uint8": 2}
_DTYPE_NAMES = {v: k for k, v in DTYPES.items()}


@dataclass
class TensorInfo:
    dtype: str
    shape: tuple


@dataclass
class Blob:
    inputs: list
    outputs: list
    rknn: bytes


def pack(blob):
    out = [struct.pack("<4sHHHH", MAGIC, VERSION, len(blob.inputs), len(blob.outputs), 0)]
    for t in [*blob.inputs, *blob.outputs]:
        out.append(struct.pack("<BBH", DTYPES[t.dtype], len(t.shape), 0))
        out.append(struct.pack(f"<{len(t.shape)}I", *t.shape))
    out.append(struct.pack("<Q", len(blob.rknn)))
    out.append(blob.rknn)
    return b"".join(out)


def unpack(data):
    magic, version, n_in, n_out, _ = struct.unpack_from("<4sHHHH", data, 0)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"not an RKNN delegate blob (magic {magic!r}, version {version})")
    offset = 12
    tensors = []
    for _ in range(n_in + n_out):
        code, ndim, _ = struct.unpack_from("<BBH", data, offset)
        shape = struct.unpack_from(f"<{ndim}I", data, offset + 4)
        tensors.append(TensorInfo(_DTYPE_NAMES[code], tuple(shape)))
        offset += 4 + 4 * ndim
    (size,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    if offset + size != len(data):
        raise ValueError(f"blob is {len(data)} bytes, header says {offset + size}")
    return Blob(tensors[:n_in], tensors[n_in:], bytes(data[offset:]))

"""The delegate blob: a small header describing the delegate's inputs and
outputs as ExecuTorch sees them, followed by the .rknn model.

All integers little-endian.

  magic    4 bytes   b"RKNB"
  version  u16       2
  n_in     u16
  n_out    u16
  reserved u16
  per input, then per output:
    dtype  u8        see DTYPES
    ndim   u8
    source u16       outputs: the .rknn output it reads (several may share
                     one, as RKNN merges identical outputs); inputs: 0
    dims   ndim x u32
  rknn_size u64
  rknn      rknn_size bytes

The runtime queries everything else (layout, quantisation) from RKNN.
"""

import struct
from dataclasses import dataclass

MAGIC = b"RKNB"
VERSION = 2
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
    output_sources: list = None  # .rknn output index per output; None: one each, in order

    def __post_init__(self):
        if self.output_sources is None:
            self.output_sources = list(range(len(self.outputs)))


def pack(blob):
    out = [struct.pack("<4sHHHH", MAGIC, VERSION, len(blob.inputs), len(blob.outputs), 0)]
    sources = [0] * len(blob.inputs) + list(blob.output_sources)
    for t, source in zip([*blob.inputs, *blob.outputs], sources):
        out.append(struct.pack("<BBH", DTYPES[t.dtype], len(t.shape), source))
        out.append(struct.pack(f"<{len(t.shape)}I", *t.shape))
    out.append(struct.pack("<Q", len(blob.rknn)))
    out.append(blob.rknn)
    return b"".join(out)


def unpack(data):
    magic, version, n_in, n_out, _ = struct.unpack_from("<4sHHHH", data, 0)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"not an RKNN delegate blob (magic {magic!r}, version {version})")
    offset = 12
    tensors, sources = [], []
    for _ in range(n_in + n_out):
        code, ndim, source = struct.unpack_from("<BBH", data, offset)
        shape = struct.unpack_from(f"<{ndim}I", data, offset + 4)
        tensors.append(TensorInfo(_DTYPE_NAMES[code], tuple(shape)))
        sources.append(source)
        offset += 4 + 4 * ndim
    (size,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    if offset + size != len(data):
        raise ValueError(f"blob is {len(data)} bytes, header says {offset + size}")
    return Blob(tensors[:n_in], tensors[n_in:], bytes(data[offset:]), sources[n_in:])

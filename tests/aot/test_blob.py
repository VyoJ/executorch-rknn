import pytest
from executorch_rknn import blob


def test_roundtrip():
    b = blob.Blob([blob.TensorInfo("float32", (1, 3, 224, 224))],
                  [blob.TensorInfo("float32", (1, 1000)), blob.TensorInfo("int8", (4,))], b"RKNN\x00model")
    assert blob.unpack(blob.pack(b)) == b


def test_outputs_can_share_a_model_output():
    t = blob.TensorInfo("float32", (1, 8))
    b = blob.Blob([t], [t, t, t], b"model", [0, 1, 0])
    assert blob.unpack(blob.pack(b)).output_sources == [0, 1, 0]


def test_rejects_other_data():
    with pytest.raises(ValueError):
        blob.unpack(b"XXXX" + bytes(16))


def test_rejects_truncated():
    data = blob.pack(blob.Blob([], [], b"abcdef"))
    with pytest.raises(ValueError):
        blob.unpack(data[:-1])

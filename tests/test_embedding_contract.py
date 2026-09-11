import numpy as np
import pytest

from src.recognition.base import BaseFaceRecognizer


def test_valid_embedding_contract_is_accepted():
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = 1.0
    validated = BaseFaceRecognizer.validate_embedding(vector)
    assert validated.shape == (512,)
    assert validated.dtype == np.float32
    assert np.isclose(np.linalg.norm(validated), 1.0)


@pytest.mark.parametrize("bad", [
    np.full(512, np.nan, dtype=np.float32),
    np.full(512, np.inf, dtype=np.float32),
    np.zeros(512, dtype=np.float32),
    np.ones(511, dtype=np.float32),
])
def test_invalid_embedding_is_rejected(bad):
    with pytest.raises(ValueError):
        BaseFaceRecognizer.validate_embedding(bad)


def test_batch_rejects_non_finite_member():
    batch = np.zeros((2, 512), dtype=np.float32)
    batch[0, 0] = 1.0
    batch[1, 0] = np.nan
    with pytest.raises(ValueError):
        BaseFaceRecognizer.validate_embedding(batch)

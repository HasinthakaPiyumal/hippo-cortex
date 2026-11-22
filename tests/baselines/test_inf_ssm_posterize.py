"""RandAugment posterize in the Inf-SSM data pipeline (baselines/inf-ssm/cl_datasets.py) must accept 0 bits.

timm's 'inc1' RandAugment asks for 4 down to 0 bits; Pillow up to 12.1 gave a black image for 0 bits,
newer Pillow raises inside the DataLoader worker. cl_datasets patches timm's op table to keep the old result.
"""
import os
import sys

import pytest
from PIL import Image, ImageOps

pytest.importorskip("timm")

INF_SSM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../baselines/inf-ssm"))
if INF_SSM_DIR not in sys.path:
    sys.path.insert(0, INF_SSM_DIR)
import cl_datasets  # noqa: E402
import timm.data.auto_augment as timm_aa  # noqa: E402


def _image(mode="RGB"):
    img = Image.new("RGB", (8, 8))
    img.putdata([(i * 4 % 256, i * 7 % 256, i * 13 % 256) for i in range(64)])
    return img.convert(mode)


def test_timm_posterize_ops_are_patched():
    for name in ("Posterize", "PosterizeIncreasing", "PosterizeOriginal"):
        assert timm_aa.NAME_TO_OP[name] is cl_datasets._posterize


@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_posterize_zero_bits_is_black(mode):
    out = cl_datasets._posterize(_image(mode), 0)
    assert out.mode == mode
    assert max(b for band in out.getextrema() for b in (band if isinstance(band, tuple) else (band,))) == 0


@pytest.mark.parametrize("bits", range(1, 9))
def test_posterize_matches_pillow(bits):
    img = _image()
    expect = img if bits >= 8 else ImageOps.posterize(img, bits)
    assert list(cl_datasets._posterize(img, bits).getdata()) == list(expect.getdata())


def test_rand_augment_op_at_max_magnitude_does_not_raise():
    op = timm_aa.AugmentOp("PosterizeIncreasing", prob=1.0, magnitude=10)  # 'inc1' at magnitude 10 -> 0 bits
    assert op.level_fn(10, op.hparams) == (0,)
    out = op(_image())
    assert max(out.getextrema()[0]) == 0

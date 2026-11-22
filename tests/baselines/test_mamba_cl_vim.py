"""CPU tests for the Vim-small backbone of the Mamba-CL baseline (baselines/mamba-cl/utils/vim_mamba.py).

The model must compute exactly what the Inf-SSM port (baselines/inf-ssm/mamba_vim.py) computes, and its
IntermReaders must feed train_eval.get_interm_tensor_dict the inputs of the parameters they name, in the
layout ModAdam multiplies the projection into.
"""
import importlib
import importlib.util
import os
import sys

import pytest
import torch
from einops import rearrange

pytest.importorskip("timm")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MCL_UTILS = os.path.join(ROOT, "baselines", "mamba-cl", "utils")
INF_SSM_DIR = os.path.join(ROOT, "baselines", "inf-ssm")


def _load_mamba_cl_utils(name="_mcl_utils_vim_test"):
    # baselines/*/utils would all be "utils"; load this one under a private package name
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, os.path.join(MCL_UTILS, "__init__.py"),
                                                      submodule_search_locations=[MCL_UTILS])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[name] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module(f"{name}.vim_mamba"), importlib.import_module(f"{name}.interm_reader")


vim_mamba, interm_reader = _load_mamba_cl_utils()
if INF_SSM_DIR not in sys.path:
    sys.path.insert(0, INF_SSM_DIR)
import mamba_vim as inf_ssm_vim  # noqa: E402

DEPTH = 2
NULL_PATTERNS = ("x_proj", "out_proj.weight", "A_log", "A_b_log")
READER_NAMES = {"interm_reader_step", "interm_reader_B", "interm_reader_C", "interm_reader_A", "interm_reader_out"}
TINY = dict(img_size=32, patch_size=16, depth=DEPTH, embed_dim=32, num_classes=5, drop_path_rate=0.)


def _models(seed=0):
    """(ours, Inf-SSM reference) with identical weights; ours already split."""
    torch.manual_seed(seed)
    ref = inf_ssm_vim.VisionMamba(**TINY).eval()
    ours = vim_mamba.VisionMamba(**TINY)
    ours.load_state_dict(ref.state_dict(), strict=True)  # same parameter names before the split
    ours.reload_params()
    return ours.eval(), ref


def _images(n=4, seed=1):
    return torch.randn(n, 3, 32, 32, generator=torch.Generator().manual_seed(seed))


def _readers(model):
    return {n: m for n, m in model.named_modules() if "interm_reader" in n and isinstance(m, interm_reader.IntermReader)}


def _collect(model, x):
    """Second-moment matrices per reader, laid out exactly as train_eval.get_interm_tensor_dict does."""
    out = {}

    def hook(module, args, _output):
        t = args[0]
        if module.module_name == "interm_reader_A":
            t = rearrange(t, "b l d -> (b d) l")
        else:
            t = rearrange(t, "b l d -> (b l) d")
        out[module.dst_param_id] = (module.module_name, t)

    handles = [m.register_forward_hook(hook) for m in _readers(model).values()]
    model.requires_grad_(False)  # collection runs fully frozen, which is when the A and B readers fire
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return out


def test_matches_the_inf_ssm_vim_after_the_split():
    ours, ref = _models()
    x = _images()
    with torch.no_grad():
        assert torch.allclose(ours(x), ref(x), atol=1e-5, rtol=1e-4)
        assert torch.allclose(ours.encode_image(x, pre_logits=True), ref.forward_features(x)[0], atol=1e-5, rtol=1e-4)


def test_forward_needs_the_split():
    model = vim_mamba.VisionMamba(**TINY).eval()
    with pytest.raises(AssertionError, match="reload_params"):
        model(_images())


def test_split_keeps_the_checkpoint_weights():
    ours, ref = _models()
    for i in range(DEPTH):
        m, r = ours.layers[i].mixer, ref.layers[i].mixer
        for sfx in ("", "_b"):
            w = torch.cat([getattr(m, f"x_proj{sfx}_{p}").weight for p in ("step", "B", "C")], dim=0)
            assert torch.equal(w, getattr(r, f"x_proj{sfx}").weight)
            assert not hasattr(m, f"x_proj{sfx}")


def test_readers_cover_every_null_space_parameter():
    model, _ = _models()
    readers = _readers(model)
    assert len(readers) == 9 * DEPTH  # step, B, C, A per direction + out_proj
    assert {r.module_name for r in readers.values()} == READER_NAMES

    params = {id(p): n for n, p in model.named_parameters() if any(s in n for s in NULL_PATTERNS)}
    assert {r.dst_param_id for r in readers.values()} == set(params)
    assert len(params) == len(readers)  # one reader per parameter, as get_interm_tensor_dict asserts


def test_default_a_log_pattern_misses_the_backward_a():
    model, _ = _models()
    a_params = [n for n, _ in model.named_parameters() if "A_log" in n]
    assert all(n.endswith(".A_log") for n in a_params) and len(a_params) == DEPTH  # hence train_eval adds A_b_log


def test_reader_inputs_match_the_axis_mod_adam_projects():
    model, _ = _models()
    params = {id(p): p for p in model.parameters()}
    collected = _collect(model, _images())
    assert len(collected) == 9 * DEPTH
    for pid, (mname, t) in collected.items():
        p = params[pid]
        if mname == "interm_reader_A":  # ModAdam: proj @ update
            assert t.shape[1] == p.shape[0], (mname, tuple(t.shape), tuple(p.shape))
        else:  # ModAdam: update @ proj
            assert t.shape[1] == p.shape[1], (mname, tuple(t.shape), tuple(p.shape))


@pytest.mark.parametrize("reader", ["interm_reader_step", "interm_reader_C", "interm_reader_out"])
def test_null_space_update_leaves_the_outputs_on_old_inputs_unchanged(reader):
    model, _ = _models()
    params = {id(p): p for p in model.parameters()}
    collected = {pid: t for pid, (mname, t) in _collect(model, _images(n=2)).items() if mname == reader}
    assert len(collected) == (2 * DEPTH if reader != "interm_reader_out" else DEPTH)

    for pid, inputs in collected.items():
        W = params[pid]
        cov = inputs.T @ inputs / inputs.shape[0]
        _, S, Vt = torch.linalg.svd(cov)
        basis = Vt[S < S.max() * 1e-6]  # directions the old inputs do not span
        assert basis.shape[0] > 0
        update = torch.randn_like(W) @ (basis.T @ basis)  # what ModAdam applies with null_eta = 1
        old_out = inputs @ W.T
        new_out = inputs @ (W + update).T
        assert (new_out - old_out).abs().max() < 1e-4 * old_out.abs().max()

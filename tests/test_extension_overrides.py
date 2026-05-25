"""Tests for ``torchada.replace_op_impl`` — the mechanism that lets a
downstream extension replace an upstream ``torch.ops.<ns>.<op>`` impl while
keeping the upstream op name in the FX graph.

These tests use a synthetic library, so they run on any platform (CPU/CUDA/MUSA).
The MUSA-specific dispatch_key translation is exercised in the dedicated MUSA
test below.
"""

import uuid

import pytest
import torch

import torchada


def _fresh_namespace() -> str:
    """Return a unique torch.library namespace so each test starts clean."""
    return f"torchada_test_{uuid.uuid4().hex[:8]}"


class TestReplaceOpImpl:
    def test_replace_op_impl_swaps_runtime_impl(self):
        """Registering a replacement at the same (op, key) tuple wins the dispatch."""
        ns = _fresh_namespace()
        op = "add_one"

        # Stand up a tiny library with a CPU impl that returns input + 1.
        lib = torch.library.Library(ns, "DEF")
        lib.define(f"{op}(Tensor x) -> Tensor")
        impl_lib = torch.library.Library(ns, "IMPL")
        impl_lib.impl(op, lambda x: x + 1, "CPU")

        op_handle = getattr(getattr(torch.ops, ns), op)
        x = torch.zeros(3)
        assert torch.equal(op_handle(x), x + 1)

        # Replace with a CPU impl that returns input - 1.
        torchada.replace_op_impl(ns, op, lambda x: x - 1, dispatch_key="CPU")
        assert torch.equal(op_handle(x), x - 1)

    def test_replace_op_impl_can_be_called_multiple_times(self):
        """Successive calls override successive impls."""
        ns = _fresh_namespace()
        op = "scale"

        lib = torch.library.Library(ns, "DEF")
        lib.define(f"{op}(Tensor x) -> Tensor")
        impl_lib = torch.library.Library(ns, "IMPL")
        impl_lib.impl(op, lambda x: x * 1.0, "CPU")

        op_handle = getattr(getattr(torch.ops, ns), op)
        x = torch.ones(2)

        torchada.replace_op_impl(ns, op, lambda x: x * 2.0, dispatch_key="CPU")
        assert torch.equal(op_handle(x), x * 2.0)

        torchada.replace_op_impl(ns, op, lambda x: x * 3.0, dispatch_key="CPU")
        assert torch.equal(op_handle(x), x * 3.0)

    def test_replace_op_impl_preserves_op_name(self):
        """The op name in ``torch.ops`` is unchanged after replacement —
        downstream code that does ``torch.ops.<ns>.<op>(...)`` works without
        edits, and Inductor pattern matchers keyed on the op handle still
        recognise it. (This is the whole point of the mechanism.)"""
        ns = _fresh_namespace()
        op = "identity"

        lib = torch.library.Library(ns, "DEF")
        lib.define(f"{op}(Tensor x) -> Tensor")
        impl_lib = torch.library.Library(ns, "IMPL")
        impl_lib.impl(op, lambda x: x, "CPU")

        op_handle_before = getattr(getattr(torch.ops, ns), op)
        torchada.replace_op_impl(ns, op, lambda x: x + 0.0, dispatch_key="CPU")
        op_handle_after = getattr(getattr(torch.ops, ns), op)

        # Same OpOverloadPacket object: the dispatcher entry is the same op,
        # only the kernel pointer behind it changed.
        assert op_handle_before is op_handle_after

    def test_list_registered_libraries(self):
        """``list_registered_libraries`` reports back what was registered."""
        ns = _fresh_namespace()
        op = "noop"

        lib = torch.library.Library(ns, "DEF")
        lib.define(f"{op}(Tensor x) -> Tensor")
        impl_lib = torch.library.Library(ns, "IMPL")
        impl_lib.impl(op, lambda x: x, "CPU")

        torchada.replace_op_impl(ns, op, lambda x: x + 0.0, dispatch_key="CPU")
        pairs = torchada.list_registered_libraries()
        assert (ns, "CPU") in pairs


@pytest.mark.musa
class TestReplaceOpImplMUSA:
    """MUSA-specific: ``dispatch_key="CUDA"`` is auto-translated to
    ``"PrivateUse1"`` by ``_patch_library_impl``, so the same call binds the
    impl to MUSA tensors without the caller knowing about ``PrivateUse1``."""

    def test_cuda_dispatch_key_translated_to_privateuse1(self):
        if not torchada.is_musa_platform():
            pytest.skip("MUSA-only test")

        ns = _fresh_namespace()
        op = "musa_add_one"

        lib = torch.library.Library(ns, "DEF")
        lib.define(f"{op}(Tensor x) -> Tensor")
        impl_lib = torch.library.Library(ns, "IMPL")
        # Caller writes "CUDA"; torchada's patched Library.impl translates to PrivateUse1.
        impl_lib.impl(op, lambda x: x + 1, "CUDA")

        x = torch.zeros(3, device="musa")
        op_handle = getattr(getattr(torch.ops, ns), op)
        assert torch.equal(op_handle(x), x + 1)

        # Replace; same translation behavior.
        torchada.replace_op_impl(ns, op, lambda x: x - 1, dispatch_key="CUDA")
        assert torch.equal(op_handle(x), x - 1)

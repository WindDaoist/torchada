# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to torchada
"""
Replace upstream ``torch.ops.<ns>.<op>`` impls with MUSA-native implementations,
keeping the upstream op name in the FX graph so downstream compile passes
that pattern-match on it (e.g. Inductor fusion patterns in vLLM under
``vllm.compilation.passes.fusion.*``) still fire and produce the upstream-named
fused result, with the runtime call routed to the MUSA-native kernel.

## Background — why this layer exists

When an upstream extension (vLLM ``_C``, torchvision, transformer_engine, …)
registers C++ kernels via ``TORCH_LIBRARY_IMPL(name, CUDA, m)``, the dispatch
key is hard-coded as ``torch::kCUDA``. torchada's build-time source rewrite
(see ``_mapping.py``) translates ``torch::kCUDA`` → ``torch::kPrivateUse1`` so
the impls fire on MUSA tensors. As a side effect, *every* such upstream op
ends up claiming the ``(_C::op, PrivateUse1)`` slot in PyTorch's dispatcher.

A downstream MUSA-native port — e.g. ``vllm-musa``'s op-perf kernels — then
cannot use ``TORCH_LIBRARY_IMPL(_C, PrivateUse1, m)`` to register its own
implementation: PyTorch's dispatcher allows at most one impl per
``(op, key)`` tuple and rejects the duplicate at library-load time with::

    Mismatch in kernel C++ signatures
      operator: _C::rms_norm_static_fp8_quant(...)
      kernel 1: ...  dispatch key: PrivateUse1  (torchada-rewritten upstream)
      kernel 2: ...  dispatch key: PrivateUse1  (downstream MUSA-native)

The naive workaround (monkey-patch ``vllm._custom_ops.<fn>`` from Python) gets
*runtime* dispatch right but Dynamo traces our ``_C_musa_ops.musa_*`` symbol
into the FX graph instead of the upstream ``_C::*`` symbol. Inductor fusion
patterns are keyed on the upstream symbol — ``qk_norm_rope_fusion.py`` has
``FUSED_QK_ROPE_OP = torch.ops._C.fused_qk_norm_rope.default`` — so the
fusion no longer matches and the model runs unfused. Measured on
``MiniMax-M2.5`` 4k/1k bf16: BS=1 wash, BS=4 -14%, BS=16 -5%, BS=64 -7%.

## What this module does

``replace_op_impl`` registers a downstream Python callable (or, preferably,
an ``OpOverload`` produced by ``torch.library.custom_op`` /
``TORCH_LIBRARY_EXPAND``) as the new impl for an existing upstream op,
using ``torch.library.Library.impl(..., allow_override=True)``. The op name
in the dispatcher stays the same (``_C::rotary_embedding`` is still
``_C::rotary_embedding``), so:

  - **Inductor fusion patterns still match.** A pattern that consumes
    ``_C.rotary_embedding`` or produces ``_C.fused_qk_norm_rope`` fires
    exactly as before.
  - **The runtime dispatch lands in the downstream's MUSA-native kernel.**
  - **Existing callers do not change.** No call-site patch, no
    monkey-patching of upstream Python wrappers.

There is a Python-dispatch tax per call when ``func`` is a Python callable
that wraps a different ``OpOverload``. For the fused case (one fused op per
N original ops), the fusion benefit dominates; for the non-fused case,
prefer to pass the downstream ``OpOverload`` directly so the dispatch stays
in C++.

## Requirements

  - PyTorch ≥ 2.4 for ``Library.impl(..., allow_override=True)``.
  - The downstream extension that owns ``func``'s symbol must be loaded
    before ``replace_op_impl`` is called.

## Example — vllm-musa wiring (illustrative)

::

    import torch
    import torch_musa  # noqa
    import vllm._C  # noqa - upstream registers _C::rotary_embedding at PrivateUse1
    import vllm_musa._C  # noqa - registers _C_musa_ops::musa_rotary_embedding
    import torchada

    # Replace the upstream PrivateUse1 impl with the MUSA-native one.
    # dispatch_key="CUDA" because torchada's _patch_library_impl
    # auto-translates it to "PrivateUse1" at registration time.
    torchada.replace_op_impl(
        "_C",
        "rotary_embedding",
        torch.ops._C_musa_ops.musa_rotary_embedding.default,
        dispatch_key="CUDA",
    )
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Library objects must outlive the registration to keep the impl alive.
# A new Library is created per (namespace, dispatch_key) pair and stashed
# here so consumers do not need to manage lifetime.
_REGISTERED_LIBS: dict[tuple[str, str], Any] = {}


def replace_op_impl(
    namespace: str,
    op_name: str,
    func: Callable[..., Any],
    *,
    dispatch_key: str = "CUDA",
) -> None:
    """Replace ``torch.ops.<namespace>.<op_name>``'s impl at ``dispatch_key``.

    Uses ``torch.library.Library.impl(..., allow_override=True)`` so any
    previously-registered impl at the same ``(op, key)`` tuple is replaced
    atomically. Keeps the op name unchanged in the FX graph so Inductor
    fusion patterns that key on the upstream op name continue to match.

    Args:
        namespace: PyTorch library namespace, e.g. ``"_C"``, ``"_moe_C"``,
            ``"_C_musa_ops"``.
        op_name: Operator name within the namespace, e.g. ``"rotary_embedding"``.
        func: New implementation. Prefer an ``OpOverload``
            (``torch.ops.<other_lib>.<op>.default``) so the runtime
            dispatch stays in C++; a Python callable also works but adds
            per-call Python overhead.
        dispatch_key: Source-level dispatch key string. Defaults to
            ``"CUDA"`` which torchada's ``_patch_library_impl`` translates
            to ``"PrivateUse1"`` at runtime — so the same call works on
            both CUDA and MUSA hosts.

    Raises:
        RuntimeError: if ``torch.library.Library.impl`` lacks the
            ``allow_override`` parameter (PyTorch < 2.4 or an unpatched build).
    """
    # Local import keeps torchada import cheap on platforms without torch.
    import torch

    # We can't reliably introspect Library.impl with `inspect.signature` here
    # because torchada's own `_patch_library_impl` wraps it with a
    # ``(self, *args, **kwargs)`` shim that hides the real parameter list.
    # Defer the availability check until the actual call below — PyTorch
    # raises ``TypeError`` if ``allow_override`` is unknown, which we
    # translate into a clearer error.

    # Reuse a Library object for the (namespace, dispatch_key) pair if one
    # exists — Library lifetime is bound to the registration, and creating
    # too many can leak entries in the dispatcher's table.
    key = (namespace, dispatch_key)
    lib = _REGISTERED_LIBS.get(key)
    if lib is None:
        lib = torch.library.Library(namespace, "IMPL")
        _REGISTERED_LIBS[key] = lib

    # The patched Library.impl (see _patch.py::_patch_library_impl) translates
    # dispatch_key="CUDA" → "PrivateUse1" on MUSA at call time. We pass the
    # source-level name so the registration works on both platforms.
    try:
        lib.impl(op_name, func, dispatch_key, allow_override=True)
    except TypeError as exc:
        if "allow_override" in str(exc):
            raise RuntimeError(
                "torchada.replace_op_impl requires PyTorch ≥ 2.4 with "
                "`torch.library.Library.impl(..., allow_override=True)`. "
                f"Underlying TypeError: {exc}"
            ) from exc
        raise

    logger.debug(
        "torchada: replaced impl %s::%s on dispatch_key=%s "
        "(translated by _patch_library_impl on MUSA)",
        namespace,
        op_name,
        dispatch_key,
    )


def list_registered_libraries() -> list[tuple[str, str]]:
    """Return the ``(namespace, dispatch_key)`` pairs that have at least one
    override registered via :func:`replace_op_impl`. Useful for diagnostics
    — does not list individual ops."""
    return list(_REGISTERED_LIBS.keys())

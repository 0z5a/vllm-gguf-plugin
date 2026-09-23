# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_gguf_plugin.weights_adapter.diffusion import integration

pytestmark = [pytest.mark.cpu]


def test_wan_cascade_scopes_dense_exclusions_before_model_init(monkeypatch):
    from vllm_omni.diffusion.model_loader import diffusers_loader

    config = {
        "method": "gguf",
        "gguf_model": {
            "transformer": "high.gguf",
            "transformer_2": "low.gguf",
        },
    }
    observed = []

    class Loader:
        quant_config = config
        od_config = SimpleNamespace(
            model_class_name="WanPipeline",
            tf_model_config={"model_type": "wan"},
            revision=None,
            dtype=torch.bfloat16,
        )
        load_config = SimpleNamespace(download_dir=None, ignore_patterns=None)
        parallel_config = SimpleNamespace(use_hsdp=False)
        counter_before_loading_weights = 0

        def load_model(self, *args, **kwargs):
            raise AssertionError("GGUF path should handle model initialization")

        def load_weights(self, model):
            raise AssertionError("GGUF path should handle weight loading")

        def _init_from_load_format(self, *args, **kwargs):
            observed.extend(self.quant_config["unquantized_modules"])
            return nn.Linear(2, 2)

        def _get_weight_sources(self, model):
            return []

        def _get_expected_parameter_names(self, model):
            return set()

        def _process_weights_after_loading(self, model, device):
            pass

    class Adapter:
        def unquantized_module_names(self):
            return ("blocks.0.attn1.to_q",)

    monkeypatch.setattr(diffusers_loader, "DiffusersPipelineLoader", Loader)
    monkeypatch.setattr(
        integration,
        "resolve_diffusion_gguf_adapters",
        lambda *args: {"transformer": Adapter(), "transformer_2": Adapter()},
    )
    monkeypatch.setattr(
        integration, "load_diffusion_gguf_weights", lambda **kwargs: set()
    )
    integration._patch_diffusers_loader()
    model = Loader().load_model(load_device="cpu")
    assert isinstance(model, nn.Linear)
    assert observed == [
        "transformer.blocks.0.attn1.to_q",
        "transformer_2.blocks.0.attn1.to_q",
    ]

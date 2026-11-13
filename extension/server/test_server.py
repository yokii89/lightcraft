"""Model-loading regression checks; no model, GPU or third-party packages required."""
import sys
import unittest
from unittest.mock import MagicMock, patch

import server


class Bf16LoadingTests(unittest.TestCase):
    def check_load(self, legacy=False):
        torch = MagicMock()
        transformers = MagicMock()
        loader = transformers.AutoModelForMultimodalLM.from_pretrained
        if legacy:
            loader.side_effect = [TypeError("use torch_dtype"), MagicMock()]
        service = server.Service(server.DEFAULT_MODEL, "test", 300, "cuda", False)
        modules = {"torch": torch, "transformers": transformers, "PIL": MagicMock()}
        with (
            patch.dict(sys.modules, modules),
            patch.object(service, "resolve_device", return_value="cuda"),
            patch.object(service, "require_free_vram"),
            patch.object(service, "resolve_model_source", return_value="local-model"),
            patch.object(service, "bound_image_pixels"),
            patch.object(service, "log_vram"),
            patch.object(service, "warmup"),
            patch.object(server, "acquire_instance_lock"),
            patch.object(server, "use_linear_patch_embed", return_value=False),
            patch.object(server, "emit"),
            patch.object(server, "log"),
            patch.object(server.os, "_exit", side_effect=AssertionError("load failed")),
        ):
            service.load()
        self.assertEqual(service.status, "ready")
        self.assertEqual(loader.call_count, 2 if legacy else 1)
        for index, call in enumerate(loader.call_args_list):
            key = "torch_dtype" if index else "dtype"
            self.assertIs(call.kwargs[key], torch.bfloat16)
            self.assertEqual(call.kwargs["device_map"], {"": 0})
            self.assertNotIn("quantization_config", call.kwargs)

    def test_cuda_load_uses_explicit_bf16_without_quantization(self):
        self.check_load()

    def test_legacy_dtype_argument_preserves_bf16(self):
        self.check_load(legacy=True)


if __name__ == "__main__":
    unittest.main()

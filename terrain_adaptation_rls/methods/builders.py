"""Lazy builders for runtime method adapters."""

from __future__ import annotations

from typing import Any


def build_runtime_method(name: str, **kwargs: Any) -> object:
    """Build a runtime method by registry name.

    Torch-heavy imports are intentionally kept inside this function so config
    validation and lightweight tooling still work in environments without torch.
    """

    if name == "linear_basis_rls":
        from terrain_adaptation_rls.methods.coefficient_adapters import (
            LinearBasisProvider,
            TorchCoefficientMethod,
        )

        provider = LinearBasisProvider(
            input_dim=kwargs.pop("input_dim", 9),
            output_dim=kwargs.pop("output_dim", 6),
            include_bias=kwargs.pop("include_bias", True),
        )
        return TorchCoefficientMethod(provider, update_rule="rls", output_dim=provider.output_dim, **kwargs)

    if name == "neuralfly_style_rls":
        from terrain_adaptation_rls.methods.coefficient_adapters import (
            NeuralFlyStyleBasisProvider,
            TorchCoefficientMethod,
        )

        provider = NeuralFlyStyleBasisProvider(
            input_dim=kwargs.pop("input_dim", 9),
            output_dim=kwargs.pop("output_dim", 6),
            n_basis=kwargs.pop("n_basis", 8),
            hidden_size=kwargs.pop("hidden_size", 128),
            n_hidden_layers=kwargs.pop("n_hidden_layers", 2),
        )
        return TorchCoefficientMethod(provider, update_rule="rls", output_dim=provider.output_dim, **kwargs)

    if name in {"fe_rls", "fe_kalman", "fe_sgd", "fe_window_ls"}:
        from terrain_adaptation_rls.methods.coefficient_adapters import (
            FunctionEncoderBasisProvider,
            TorchCoefficientMethod,
        )

        try:
            model = kwargs.pop("model")
        except KeyError as exc:
            raise ValueError(f"{name} requires a loaded FunctionEncoder model") from exc

        update_rules = {
            "fe_rls": "rls",
            "fe_kalman": "kalman",
            "fe_sgd": "sgd",
            "fe_window_ls": "window_ls",
        }
        provider = FunctionEncoderBasisProvider(model)
        return TorchCoefficientMethod(provider, update_rule=update_rules[name], **kwargs)

    raise NotImplementedError(f"No runtime builder is available for '{name}'")

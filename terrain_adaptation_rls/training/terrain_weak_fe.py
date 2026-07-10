"""Pure weak-form FE training for Phoenix-shaped terrain data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import torch

from terrain_adaptation_rls.configuration import ExperimentConfig
from terrain_adaptation_rls.data.load_data import load_scenes
from terrain_adaptation_rls.evaluation.diagnostic_plots import write_fe_diagnostics
from terrain_adaptation_rls.evaluation.vdp_weak_fe import solve_coefficients
from terrain_adaptation_rls.evaluation.vdp_weak_fe import weak_system
from terrain_adaptation_rls.models.function_encoder import create_model


def run_terrain_weak_fe_training(
    config: ExperimentConfig,
    *,
    device: torch.device | str = "cpu",
    max_steps: int | None = None,
    artifact_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train FE basis functions using only weak-form trajectory residuals."""

    if config.platform is None:
        raise ValueError("Weak FE training requires config.platform")
    train_scenes = [str(scene) for scene in config.data.get("train_scenes", ())]
    validation_scenes = [str(scene) for scene in config.data.get("validation_scenes", ())]
    if not train_scenes:
        raise ValueError("Weak FE training requires data.train_scenes")

    torch.manual_seed(config.seed)
    device = torch.device(device)

    model_config = config.model
    n_basis = int(model_config.get("n_basis", 16))
    hidden_size = int(model_config.get("hidden_size", 256))
    model = create_model(device, n_basis=n_basis, hidden_size=hidden_size)

    training = config.training
    configured_steps = int(training.get("steps", 1000))
    steps = configured_steps if max_steps is None else min(configured_steps, max_steps)
    batch_size = int(training.get("batch_size", 8))
    learning_rate = float(training.get("learning_rate", 5e-4))
    ridge = float(training.get("weak_ridge", training.get("ridge", 1e-4)))
    norm_weight = float(training.get("norm_weight", 1e-3))
    coeff_weight = float(training.get("coeff_weight", 1e-5))
    gradient_clip_norm = float(training.get("gradient_clip_norm", 1.0))
    dt = float(training.get("weak_dt", 0.1))
    window = int(training.get("weak_window", 41))
    powers = tuple(int(value) for value in training.get("weak_powers", (4, 6, 8, 10)))
    example_starts = tuple(
        int(value) for value in training.get("weak_example_starts", (0, 15, 30, 45))
    )
    query_starts = tuple(
        int(value) for value in training.get("weak_query_starts", (60, 75, 90, 105))
    )
    eval_example_starts = tuple(
        int(value)
        for value in training.get("weak_eval_example_starts", example_starts + query_starts[:1])
    )
    eval_query_start = int(training.get("weak_eval_query_start", max(query_starts)))
    eval_query_points = int(config.evaluation.get("max_eval_points", 1024))
    log_interval = int(training.get("log_interval", 0))

    train_data = load_scenes(train_scenes, config.platform)
    train_windows = [
        _prepared_scene_tensors(inputs, targets, device=device)
        for inputs, targets in train_data.values()
    ]
    validation_data = load_scenes(validation_scenes, config.platform) if validation_scenes else {}

    segment_length = _required_segment_length(
        window=window,
        example_starts=example_starts,
        query_starts=query_starts,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses: list[float] = []
    weak_losses: list[float] = []
    norm_losses: list[float] = []
    coeff_losses: list[float] = []
    for step in range(1, steps + 1):
        observed = _sample_segments(
            train_windows,
            batch_size=batch_size,
            segment_length=segment_length,
            device=device,
        )
        basis = evaluate_raw_basis_functions(model, observed)
        context_design, context_target = weak_system(
            observed[..., :6],
            basis,
            dt=dt,
            starts=example_starts,
            window=window,
            powers=powers,
        )
        coefficients, gram = solve_coefficients(
            context_design,
            context_target,
            regularization=ridge,
        )
        query_design, query_target = weak_system(
            observed[..., :6],
            basis,
            dt=dt,
            starts=query_starts,
            window=window,
            powers=powers,
        )
        prediction = torch.einsum("bnk,bk->bn", query_design, coefficients)
        weak_loss = torch.nn.functional.mse_loss(prediction, query_target)
        gram_diag = torch.diagonal(gram.mean(dim=0))
        norm_loss = ((gram_diag / gram_diag.detach().mean().clamp_min(1e-6) - 1.0) ** 2).mean()
        coeff_loss = (coefficients**2).mean()
        loss = weak_loss + norm_weight * norm_loss + coeff_weight * coeff_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradient_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        weak_losses.append(float(weak_loss.detach().cpu()))
        norm_losses.append(float(norm_loss.detach().cpu()))
        coeff_losses.append(float(coeff_loss.detach().cpu()))

        if log_interval > 0 and (step == 1 or step == steps or step % log_interval == 0):
            print(
                f"step {step}/{steps} loss={losses[-1]:.6g} "
                f"weak={weak_losses[-1]:.6g} norm={norm_losses[-1]:.6g} "
                f"coeff={coeff_losses[-1]:.6g}",
                flush=True,
            )

    metrics: dict[str, object] = {
        "family": "function_encoder",
        "training_mode": "pure_weak_form",
        "device": str(device),
        "platform": config.platform,
        "train_scenes": train_scenes,
        "validation_scenes": validation_scenes,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_basis": n_basis,
        "hidden_size": hidden_size,
        "weak_dt": dt,
        "weak_window": window,
        "weak_powers": list(powers),
        "weak_example_starts": list(example_starts),
        "weak_query_starts": list(query_starts),
        "weak_eval_example_starts": list(eval_example_starts),
        "weak_eval_query_start": eval_query_start,
        "weak_eval_query_points": eval_query_points,
        "ridge": ridge,
        "norm_weight": norm_weight,
        "coeff_weight": coeff_weight,
        "gradient_clip_norm": gradient_clip_norm,
        "losses": losses,
        "weak_losses": weak_losses,
        "norm_losses": norm_losses,
        "coeff_losses": coeff_losses,
        "final_loss": losses[-1] if losses else None,
    }

    if artifact_dir is not None:
        artifact_path = Path(artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), artifact_path / "function_encoder_model.pth")
        write_training_plot(artifact_path, metrics)
        if validation_data:
            scene, (inputs, targets) = next(iter(validation_data.items()))
            validation_batch = build_weak_validation_batch(
                model,
                inputs,
                targets,
                dt=dt,
                window=window,
                powers=powers,
                example_starts=eval_example_starts,
                query_start=eval_query_start,
                query_points=eval_query_points,
                ridge=ridge,
                device=device,
            )
            write_validation_artifacts(
                artifact_path,
                model,
                validation_batch,
                scene=scene,
            )
        (artifact_path / "weak_summary.json").write_text(json.dumps(metrics, indent=2) + "\n")

    return metrics


def evaluate_raw_basis_functions(model: torch.nn.Module, xs: torch.Tensor) -> torch.Tensor:
    try:
        basis_modules = model.basis_functions.basis_functions
    except AttributeError as exc:
        raise TypeError("model does not expose FunctionEncoder basis modules") from exc
    time = torch.zeros(xs.shape[:-1], dtype=xs.dtype, device=xs.device)
    values = [basis.ode_func(time, xs) for basis in basis_modules]
    return torch.stack(values, dim=-1)


def build_weak_validation_batch(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    dt: float,
    window: int,
    powers: Sequence[int],
    example_starts: Sequence[int],
    query_start: int,
    query_points: int,
    ridge: float,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    xs_all, dt_all, ys_all = _prepared_scene_tensors(inputs, targets, device=device)
    required = max(max(example_starts) + window, query_start + query_points)
    if xs_all.shape[0] < required:
        raise ValueError("validation scene is too short for requested weak validation windows")
    observed = xs_all[:required].unsqueeze(0)
    basis = evaluate_raw_basis_functions(model, observed)
    context_design, context_target = weak_system(
        observed[..., :6],
        basis,
        dt=dt,
        starts=example_starts,
        window=window,
        powers=powers,
    )
    coefficients, _ = solve_coefficients(context_design, context_target, regularization=ridge)
    query_indices = torch.arange(query_start, query_start + query_points, device=device)
    query_xs = xs_all[query_indices].unsqueeze(0)
    query_dt = dt_all[query_indices].unsqueeze(0)
    query_ys = ys_all[query_indices].unsqueeze(0)
    return query_xs, query_dt, query_ys, query_xs, query_dt, query_ys, coefficients


@torch.no_grad()
def write_validation_artifacts(
    artifact_dir: Path,
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, ...],
    *,
    scene: str,
) -> None:
    query_xs, query_dt, target, _, _, _, coefficients = batch
    prediction = model((query_xs, query_dt), coefficients=coefficients)
    error = torch.linalg.norm(prediction - target, dim=-1)
    time = torch.cumsum(query_dt.squeeze(0), dim=0)

    csv_path = artifact_dir / "validation_predictions.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["scene", "index", "time", "error"]
        fieldnames += [f"target_{idx}" for idx in range(target.shape[-1])]
        fieldnames += [f"prediction_{idx}" for idx in range(prediction.shape[-1])]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        target_cpu = target.squeeze(0).detach().cpu()
        prediction_cpu = prediction.squeeze(0).detach().cpu()
        error_cpu = error.squeeze(0).detach().cpu()
        time_cpu = time.detach().cpu()
        for idx in range(target_cpu.shape[0]):
            row = {
                "scene": scene,
                "index": idx,
                "time": float(time_cpu[idx]),
                "error": float(error_cpu[idx]),
            }
            row.update({f"target_{dim}": float(target_cpu[idx, dim]) for dim in range(6)})
            row.update(
                {f"prediction_{dim}": float(prediction_cpu[idx, dim]) for dim in range(6)}
            )
            writer.writerow(row)

    write_fe_diagnostics(
        artifact_dir,
        model=model,
        batch=batch[:6],
        prediction=prediction,
        scene=scene,
    )


def write_training_plot(artifact_dir: Path, metrics: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in (
        ("losses", "total"),
        ("weak_losses", "weak"),
        ("norm_losses", "gram norm"),
        ("coeff_losses", "coeff"),
    ):
        values = metrics.get(key, [])
        if values:
            ax.plot(values, label=label, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(artifact_dir / "training_curve.png", dpi=160)
    plt.close(fig)


def _prepared_scene_tensors(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs_all = inputs[:, 1:].to(device)
    dt_all = (targets[:, 0] - inputs[:, 0]).to(device)
    ys_all = targets[:, 1:].to(device) - xs_all[:, :6]
    return xs_all, dt_all, ys_all


def _required_segment_length(
    *,
    window: int,
    example_starts: Sequence[int],
    query_starts: Sequence[int],
) -> int:
    return max(max(example_starts) + window, max(query_starts) + window)


def _sample_segments(
    scenes: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    batch_size: int,
    segment_length: int,
    device: torch.device,
) -> torch.Tensor:
    segments: list[torch.Tensor] = []
    for _ in range(batch_size):
        scene_index = torch.randint(0, len(scenes), (1,), device=device).item()
        xs_all, _, _ = scenes[scene_index]
        if xs_all.shape[0] < segment_length:
            raise ValueError("scene is too short for requested weak-form segment")
        start = torch.randint(0, xs_all.shape[0] - segment_length + 1, (1,), device=device).item()
        segments.append(xs_all[start : start + segment_length])
    return torch.stack(segments)

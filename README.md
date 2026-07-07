# Zero to Autonomy in Real-Time: Online Adaptation of Dynamics in Unstructured Environments

Online terrain adaptation via Function Encoders with recursive least-squares
coefficient updates.

## Current Workflow

The rewrite path is config-driven and keeps generated artifacts under
`outputs/`. Use the devcontainer when possible so training dependencies do not
pollute the host Python install.

Check GPU availability before launching CUDA jobs:

```bash
nvidia-smi
```

Train the canonical scaled Function Encoder config:

```bash
python3 -m terrain_adaptation_rls.experiments.train_fe --device cuda:0
```

For a quick debug run without editing the config:

```bash
python3 -m terrain_adaptation_rls.experiments.train_fe \
  --device cuda:0 \
  --max-steps 100 \
  --run-name fe_debug
```

Validate the config without creating artifacts:

```bash
python3 -m terrain_adaptation_rls.experiments.train_fe --dry-run
```

The default config is `configs/train/warty_fe_scaled.json`. Smaller debug
configs live beside it in `configs/train/`.

## Training Artifacts

Training runs write to `outputs/train/<timestamp>_<run-name>/`. Useful files
include:

- `function_encoder_model.pth`
- `training_metrics.json`
- `training_curve.png`
- `validation_components.png`
- `validation_trajectory_snapshot.png`
- `validation_delta_scale.png`
- `phase_streamplot.png`
- `basis_streamplots.png`
- `conditioning_summary.json`
- `trajectory_summary.json`

The FE model contract is Phoenix-shaped: `(xs, dt) -> delta_state`. Static FE
evaluation computes coefficients from example points, while FE-RLS starts from
an online coefficient state and applies predict-before-update semantics.

## FE-RLS Streaming Diagnostics

After training an FE model, stream a scene through online FE-RLS:

```bash
python3 -m terrain_adaptation_rls.experiments.eval_fe_rls \
  --train-run-dir outputs/train/<timestamp>_<run-name> \
  --scene scene1 \
  --device cuda:0 \
  --run-name fe_rls_scene1
```

This writes to `outputs/eval/<timestamp>_<run-name>/`. Useful files include
`streaming_error.png`, `streaming_components.png`, `streaming_delta_scale.png`,
`streaming_trajectory.png`, `rls_coefficients.png`, `summary.json`, and
`streaming_predictions.csv`.

## Tests

Run lightweight host tests:

```bash
python3 scripts/test_local.py
```

Run the full test suite inside the devcontainer:

```bash
docker exec -w /workspaces/terrain-adaptation-rls busy_cohen python3 scripts/test_local.py
```

## Legacy Scripts

The original student-written scripts remain in place as reference behavior, but
they are no longer the preferred training/evaluation surface. See
`legacy/README.md` and `legacy/script_audit.json` for the transition map.

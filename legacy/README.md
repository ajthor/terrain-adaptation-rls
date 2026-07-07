# Legacy Script Audit

The original scripts remain in place while the rewrite is built. Treat them as
reference behavior to replicate or intentionally correct, not as the architecture
for new work.

The top-level README now documents the config-driven FE workflow. Use this
legacy note when you need to compare against the old hardcoded scripts or recover
their plotting conventions.

Legacy entrypoints still present in `terrain_adaptation_rls/` include:

- `train.py` and `train_all.sh` for the original seed/model sweeps.
- `plot_error_k_step.py` and `plot_error_k_step_with_adaptation.py` for rollout
  error plots.
- `eval_error_over_seeds_sequntial.py` and related `plot_error_*` scripts for
  sequential scene error plots.
- `eval_coeff_evolution_over_sequential.py` and `plot_coeffs_and_error_animated.py`
  for FE-RLS coefficient evolution.

Known audit points:

- `train_all.sh` passes `--grad`, while `terrain_adaptation_rls/train.py` expects
  `--gradsteps`.
- Several eval/plot scripts hardcode platform, scene, seeds, and adaptation
  hyperparameters.
- FE-RLS currently updates coefficients from basis outputs while the FE bases may
  be NeuralODE/RK4-integrated vector fields. The rewrite must make the discrete
  increment versus derivative target explicit.
- MAML device handling is embedded in the loss function and should be centralized.
- Plot scripts often load models and recompute predictions; new analysis should
  read saved eval artifacts instead.

See `legacy/script_audit.json` for the script-by-script transition map.

# Legacy Script Audit

The original scripts remain in place while the rewrite is built. Treat them as
reference behavior to replicate or intentionally correct, not as the architecture
for new work.

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

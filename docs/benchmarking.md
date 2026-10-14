# Benchmark protocol

1. Fix dataset generation, train/validation split, preprocessing and objective normalization. Use multiple seeds. For real datasets, document provenance and checksums, label mapping, missing-value treatment and feature scaling.
2. Run a portable Release build first. Record compiler version, build flags, CPU model, available RAM, dependency revisions and thread configuration beside results. The runner records runtime metadata, but compiler and hardware details must be added by the experiment owner.
3. Check gradients and compare solutions with a trusted independently formulated objective. Distinguish a tolerance stop, epoch exhaustion and a failed line search.
4. Tune solver hyperparameters on a separate validation protocol with equivalent search budgets. Defaults are baselines, not equally tuned configurations.
5. Compare convergence traces versus elapsed time, not just iterations. Report full-gradient norm, penalized training objective and unpenalized held-out log loss. Validation accuracy alone hides calibration and optimization differences.
6. Separate Python conversion costs, C++ optimization and prediction. Add warm-up and repeated runs for throughput claims; control CPU frequency, background load and memory pressure. Measure peak RSS externally when investigating memory. A single synthetic run is only a smoke benchmark.

The runner saves raw per-seed metrics and traces; it intentionally does not invent a winner or a speedup baseline. SGD/Adam may oscillate with constant step sizes. Large sparse feature spaces currently incur dense O(p) work per mini-batch, so feature count is as important as nonzero count.

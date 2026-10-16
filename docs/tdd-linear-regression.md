# Linear regression TDD record — 0.3.0

The linear API and implementation were developed through observed failing tests, independent-reference checks, and regression repairs. Tests were not postponed until the benchmark/report was complete.

## RED: absent functionality

Before adding the implementation:

~~~bash
python -m pytest -q tests/test_linear.py tests/test_regression_analysis.py
g++ -std=c++17 -fsyntax-only tests/linear_test.cpp -Iinclude -I/path/to/abseil
~~~

The Python run failed collection with missing LinearRegression/ElasticNet imports. The C++ check failed because logitlab/linear_regression.hpp did not exist.

Tests specified NumPy least-squares/augmented Ridge solutions, weighted dense/CSR equality, L1 soft-threshold closed forms, SciPy bounded least squares with fixed coordinates eliminated, independent positive Elastic Net solutions, and invalid/refit behavior.

## GREEN and precision repair

The first implementation exposed a FISTA precision failure at tolerance=1e-9. A backtracking allowance based on the full objective allowed steps whose tiny loss changes were hidden by rounding. The implementation now checks the quadratic curvature condition through gradient differences and uses monotone restart; independent coefficient/prediction references then passed.

An initial test assumption that Ridge must exceed R²=0.98 was also corrected: the independently computed regularized optimum had R²≈0.9688. The test now compares predictive R² to that reference rather than requiring an arbitrary score.

The initial solution/inference group then passed **59 tests**.

## RED/GREEN: numerical edge regressions

Additional tests first produced five failures:

1. Zero-weight observations could contaminate CG/HVP and proximal evaluation through 0×∞ when a finite feature became an overflowing score.
2. Centered variance rounding for a constant seven-element target produced a very negative R² instead of the documented finite policy.
3. A large intercept with ordinary residual noise was mistaken for zero residual variance by a target-magnitude-based cutoff.

Zero-weight rows now skip numerical kernels; constant targets are detected exactly; inference requires positive finite residual variance without rejecting valid large-offset data. The corresponding group then passed **66 tests**.

A separate failing test found that Ridge(l1=...) and Lasso(l2=...) could silently become mixed-penalty models. Named constructors now reject conflicting regularizers.

Final integration added constant-target fitting tests, including zero-weight exceptions and targets of magnitude 1e308. The normalized intercept sum rounded a constant response and gave finite R²=0 for a nearly exact fit. Initialization now preserves identical positive-weight targets exactly; the final performance study was rerun against this compiled implementation.

## RED/GREEN: benchmark and reporting

The benchmark tests initially failed because benchmarks.linear_suite was absent. They specify deterministic shared splits, train-only RMS, transformed coefficient constraints, verified independent references, and complete CLI artifacts. The report test initially failed because benchmarks.linear_report was absent; it now rejects incomplete experiment coverage.

Penalty selection uses validation predictions from independent NumPy/SciPy/scikit-learn fits. Test labels do not select penalties. Core timing is measured only after a model's objective has been fixed, so solver speed does not choose a different model.

## Validation boundary

Python coverage measures wrapper/statistics code, not C++ numerical kernels. Native CTest checks gradients, weighted optimization, constraints and invalid C++ inputs. Release, sanitizer and source-to-wheel checks cover the compiled paths. The measured research report includes unconverged results; passing tests do not imply every finite-budget experiment reaches its target.

~~~bash
python -m pytest -q --cov=logitlab --cov-report=term-missing --cov-fail-under=95
cmake --build build/release --parallel 2
ctest --test-dir build/release --output-on-failure
~~~

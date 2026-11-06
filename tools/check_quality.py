"""Enforce per-function complexity budgets for the modern runtime and bindings."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import lizard
from radon.complexity import cc_visit
from radon.visitors import Class


def cpp_functions(root):
    records = []
    for directory in ("src", "include", "bindings"):
        for file in sorted((root / directory).rglob("*")):
            if file.suffix not in (".cpp", ".hpp"):
                continue
            for function in lizard.analyze_file(str(file)).function_list:
                records.append(
                    dict(
                        file=str(file.relative_to(root)),
                        name=function.name,
                        complexity=function.cyclomatic_complexity,
                        lines=function.length,
                    )
                )
    return records


def python_functions(root):
    records = []
    for file in sorted((root / "python/logitlab").glob("*.py")):
        for block in cc_visit(file.read_text()):
            functions = block.methods if isinstance(block, Class) else [block]
            for function in functions:
                prefix = block.name + "." if isinstance(block, Class) else ""
                records.append(
                    dict(
                        file=str(file.relative_to(root)),
                        name=prefix + function.name,
                        complexity=function.complexity,
                        lines=function.endline - function.lineno + 1,
                    )
                )
    return records


def audit(root, budget):
    cpp, python = cpp_functions(root), python_functions(root)
    return dict(
        scope="Modern C++ core/public headers/bindings and Python runtime; historical code and research reports excluded",
        tools=dict(lizard=version("lizard"), radon=version("radon")),
        budget=budget,
        cpp=cpp,
        python=python,
        max_cpp=max(r["complexity"] for r in cpp),
        max_python=max(r["complexity"] for r in python),
        violations=[r for r in cpp + python if r["complexity"] > budget],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, default=Path("build/quality.json"))
    parser.add_argument("--max-complexity", type=int, default=15)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.max_complexity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"C++ max={result['max_cpp']}; Python max={result['max_python']}; budget={args.max_complexity}"
    )
    if result["violations"]:
        for row in result["violations"]:
            print(row)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

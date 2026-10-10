# Contributing to NexRT

Run the code checks from the repository root:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

The hook configuration pins Ruff and clang-format versions. Hooks check Python
code and format Python, C++, and CUDA sources. Review any formatting changes and
run the checks again before committing.

Use four spaces for indentation and a 100-column formatting target. Keep changes
focused and include relevant tests for behavior changes.

## Naming

- Python: `snake_case` for functions, attributes, and variables; `PascalCase` for
  classes; `UPPER_SNAKE_CASE` for module constants.
- C++/CUDA: `PascalCase` for types and exported functions; `PascalCaseKernel` for
  kernels; `snake_case` for variables and low-level helpers; `kCamelCase` for
  constants; `UPPER_SNAKE_CASE` for macros.
- Files and namespaces use `snake_case`. Preserve existing kernel-family
  namespaces and established abbreviations such as `CP`, `FP8`, and `GDN`.
- Keep conventional mathematical symbols where they clarify tensor operations.
  Preserve external API names, checkpoint keys, and C ABI symbols unless the
  change explicitly updates their callers.

Keep the existing Apache-2.0 copyright and license header in source files, and
include it in new source files.

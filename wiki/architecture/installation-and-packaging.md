# Installation and Packaging

> Read installation instructions separately from declared package requirements. · Status: confirmed
> Up: [Architecture](index.md)

## Summary

The README documents installation with pip in a virtual environment.
`setup.py` declares Python `>=3.6` and reads installation dependencies from
`requirements.txt`. Those declarations describe requested packages and
metadata; they do not establish successful installation, import, or numerical
evaluation on a particular Python and dependency combination.

## Detail

### Declared Requirements and README Guidance

`setup.py` checks for Python 3.6 or newer and passes `python_requires=">=3.6"`
to `setup`. `read_requirements_file` reads `requirements.txt` with
`splitlines()`, and the resulting list becomes `install_requires`. That file
requests `sympy`, `black`, and `nrpy` without version constraints. Black is
therefore included in the declared installation dependencies.

Claim evidence:
- Claim: Package setup declares Python `>=3.6` and unpinned installation dependencies `sympy`, `black`, and `nrpy`; these are declarations rather than proof that every dependency combination installs or runs.
- Role: descriptive behavior
- Deciding authority: [setup.py](../../setup.py), `check_python_version`, `read_requirements_file`, and the `setup` call's `python_requires` and `install_requires`; [requirements.txt](../../requirements.txt), dependency entries.
- Corroboration: none available for successful execution across dependency combinations; package declarations alone cannot supply that evidence.

The README's `Installation instructions` gives Python 3.6+, SymPy 1.2+,
and NRPy+ 2.0+ as prerequisites. Its package guidance is more specific about
lower bounds than the unpinned dependency entries, and it does not list
Black there. Use `requirements.txt` and `setup.py` to answer what pip is
asked to install; use the README to identify the documented prerequisites.
Do not describe those README lower bounds as enforced dependency pins or as
verified runtime compatibility.

Claim evidence:
- Claim: The README recommends Python 3.6+, SymPy 1.2+, and NRPy+ 2.0+, while the installation metadata requests unpinned `sympy`, `black`, and `nrpy`; the README lower bounds are not encoded in `install_requires`.
- Role: descriptive behavior
- Deciding authority: [setup.py](../../setup.py), `read_requirements_file` and `setup(install_requires=requirements)`; [requirements.txt](../../requirements.txt), dependency entries; [README.md](../../README.md), `## Installation instructions`, for the stated guidance.
- Corroboration: none available; the comparison is between the defining package inputs and their accompanying installation instructions, not a compatibility test.

### Installation Commands

For installation from the package index, the README supplies this Bash
sequence and then invokes the command-line module:

```bash
python3 -m venv nrpyvirtualenv
source nrpyvirtualenv/bin/activate
pip install nrpypn
python3 -m nrpypn.quasicirc
```

Claim evidence:
- Claim: The documented package-index route creates and activates a virtual environment, installs `nrpypn` with pip, and invokes `nrpypn.quasicirc`; this is an instruction sequence, not recorded evidence of a successful install or calculation.
- Role: descriptive behavior
- Deciding authority: [README.md](../../README.md), `### Quick Install from the Command Line (Bash Shell)`.
- Corroboration: [quasicirc.py](../../nrpypn/quasicirc.py), `if __name__ == "__main__"`, calls `main` for module execution.

For installation from a local source checkout, the `setup.py` module
docstring specifies `pip install .` from the directory containing
`setup.py`. This selects the local source rather than the package-index
release. Resolution of dependencies and execution of the installed package
still require separate checks.

Claim evidence:
- Claim: The source-checkout installation instruction is `pip install .` from the repository root containing `setup.py`; it does not itself demonstrate dependency compatibility or successful numerical evaluation.
- Role: descriptive behavior
- Deciding authority: [setup.py](../../setup.py), installation instruction in the module docstring and the guarded `setup` call.
- Corroboration: [MANIFEST.in](../../MANIFEST.in), `include requirements.txt` and `include nrpypn/release.txt`, identifies source-distribution inputs read by setup.

### Package Metadata and Included Data

The `setup` call names the package `nrpypn`, discovers packages with
`find_packages()`, and obtains its version using
`get_nrpypn_version(dir_setup)`. That function reads `nrpypn/release.txt`
and extracts a line beginning `version =`; it raises `ValueError` if no
such line is found. `package_data` requests the `nrpypn/py.typed` marker.
`MANIFEST.in` explicitly includes `requirements.txt` and
`nrpypn/release.txt` in the source distribution. These statements describe
the build inputs and inclusion rules, not inspected distribution contents.

Claim evidence:
- Claim: Setup discovers packages, reads version information from `nrpypn/release.txt`, and requests inclusion of `py.typed`; the manifest explicitly includes the requirements and release files in the source distribution, without proving the contents of a built distribution.
- Role: descriptive behavior
- Deciding authority: [setup.py](../../setup.py), `get_nrpypn_version` and the `setup` call's `name`, `packages`, `version`, and `package_data`; [MANIFEST.in](../../MANIFEST.in), `include` directives.
- Corroboration: [release.txt](../../nrpypn/release.txt), `version` assignment; [py.typed](../../nrpypn/py.typed), package typing marker.

### Module Invocation and Imports

`setup.py` does not declare a console-script entry point. The documented
invocation is `python3 -m nrpypn.quasicirc`. The package's `__init__.py` is
empty, so import the numerical function from
`nrpypn.eval_p_t_and_p_r`, as shown in
[Numerical Evaluation](../usage/numerical-evaluation.md).

Claim evidence:
- Claim: Setup declares no console-script entry point, `nrpypn.__init__` re-exports no API, and the documented CLI is the `nrpypn.quasicirc` module.
- Role: descriptive behavior
- Deciding authority: [setup.py](../../setup.py), complete `setup` keyword list; [__init__.py](../../nrpypn/__init__.py), empty package initializer; [README.md](../../README.md), `### Quick Install from the Command Line (Bash Shell)`.
- Corroboration: [quasicirc.py](../../nrpypn/quasicirc.py), import of `eval__P_t__and__P_r` and its `__main__` block.

## Sources

- [README.md](../../README.md) — `## Installation instructions` and `### Quick Install from the Command Line (Bash Shell)`.
- [setup.py](../../setup.py) — installation module docstring, `check_python_version`, `read_requirements_file`, `get_nrpypn_version`, and `setup` arguments.
- [requirements.txt](../../requirements.txt) — unpinned dependency entries.
- [MANIFEST.in](../../MANIFEST.in) — explicit `include` directives.
- [release.txt](../../nrpypn/release.txt) — `version` assignment read by setup.
- [py.typed](../../nrpypn/py.typed) — typing marker requested by `package_data`.
- [__init__.py](../../nrpypn/__init__.py) — empty package initializer.
- [quasicirc.py](../../nrpypn/quasicirc.py) — function import and `__main__` block.

## See Also

- Parent: [Architecture](index.md)
- See also: [Calculation Overview](overview.md)
- Example: [Numerical Evaluation](../usage/numerical-evaluation.md)
- Example: [Command Line and TwoPunctures](../usage/command-line-and-twopunctures.md)

# NRPy Expression-Validation Definitions

Verbatim selected definitions from NRPy's
`nrpy/validate_expressions/validate_expressions.py`, preserved from the reference
checkout. Upstream source: https://github.com/nrpy/nrpy/blob/main/nrpy/validate_expressions/validate_expressions.py

These excerpts preserve the helper implementation for cited behavior. They are
not a standalone module and do not describe every release of the unpinned
`nrpy` dependency. The defining module supplies imports and the comparison
function referenced below.

## compare_or_generate_trusted_results

```python
def compare_or_generate_trusted_results(
    os_path_abspath: str,
    os_getcwd: str,
    trusted_file_basename: str,
    results_dict: Dict[Any, Union[mpc, mpf]],
) -> None:
    """
    Compare the results dictionary against a trusted dictionary or (if trusted dict does not exist) generate a new one.

    :param os_path_abspath: The absolute path to the current module. Use os.path.abspath(__file__)
    :param os_getcwd: The current working directory. Use os.getcwd()
    :param trusted_file_basename: The base name for the trusted file.
    :param results_dict: The dictionary containing the results to compare against trusted.
    """
    # Calculate the relative path to the directory where the trusted file will be stored
    trusted_file_relpath = Path(os_path_abspath).relative_to(os_getcwd)
    outdir = trusted_file_relpath.parent / "tests"

    # Determine the output directory and create it if it doesn't exist
    if not outdir.is_dir():
        outdir.mkdir(parents=True, exist_ok=True)

    outfile_path = outdir / f"{trusted_file_basename}.py"

    if outfile_path.exists():
        compare_against_trusted(
            os_path_abspath,
            os_getcwd,
            trusted_file_basename,
            results_dict,
        )
    else:
        output_trusted(
            os_path_abspath,
            os_getcwd,
            trusted_file_basename,
            results_dict,
        )
```

## output_trusted

```python
def output_trusted(
    os_path_abspath: str,
    os_getcwd: str,
    trusted_file_basename: str,
    results_dict: Dict[Any, Union[mpc, mpf]],
) -> None:
    """
    Write a trusted dictionary to a file with appropriate import headers.

    :param os_path_abspath: The absolute path to the current module. Use os.path.abspath(__file__)
    :param os_getcwd: The current working directory. Use os.getcwd()
    :param trusted_file_basename: The base name for the trusted file.
    :param results_dict: The dictionary containing the results to output as trusted.

    :raises RuntimeError: If an error occurs during file writing.
    """
    output_str = f"trusted_dict = {results_dict}\n"

    # Determine necessary import headers based on the presence of mpf and mpc in the output string
    if "mpc(" in output_str and "mpf(" in output_str:
        header = "from mpmath import mpc, mpf  # type: ignore\n"
    elif "mpf(" in output_str:
        header = "from mpmath import mpf  # type: ignore\n"
    elif "mpc(" in output_str:
        header = "from mpmath import mpc  # type: ignore\n"
    else:
        header = ""

    # Calculate the relative path to the directory where the trusted file will be stored
    trusted_file_relpath = Path(os_path_abspath).relative_to(os_getcwd)
    outdir = trusted_file_relpath.parent / "tests"

    # Determine the output directory and create it if it doesn't exist
    outdir.mkdir(parents=True, exist_ok=True)

    outfile_path = outdir / f"{trusted_file_basename}.py"

    try:
        with outfile_path.open("w", encoding="utf-8") as f:
            formatted_str = black.format_str(header + output_str, mode=black.FileMode())
            f.write(formatted_str)
    except Exception as e:
        raise RuntimeError(f"Error writing to file: {e}") from e

    print(f"*** Outputted trusted dictionary to {outfile_path} . ***")
```


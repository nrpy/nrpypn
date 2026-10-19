# NRPy Expression-Sampling Definition

Verbatim selected definition from the reference checkout's
`nrpy/validate_expressions/validate_expressions.py`. Upstream source:
https://github.com/nrpy/nrpy/blob/main/nrpy/validate_expressions/validate_expressions.py

This excerpt preserves numerical sample selection, not every release of the
unpinned `nrpy` dependency. It is not a standalone module; the defining module
supplies its imports. The separate expression-validation excerpt remains
unchanged.

## convert_free_symbols_set_to_mpf_dict

```python
def convert_free_symbols_set_to_mpf_dict(
    free_symbols_set: Set[sp.Symbol],
    fixed_mpfs_for_free_symbols: bool = False,
    hex_offset: int = 0,
) -> Dict[sp.Symbol, mpf]:
    """
    Convert a set of free symbols into a dictionary with mpf values.

    :param free_symbols_set: Set of free symbols to be converted.
    :param fixed_mpfs_for_free_symbols: Flag to indicate if the mpf values should be fixed.
    :param hex_offset: Offset for random number; zero by default.

    :return: Dictionary with free symbols as keys and their corresponding mpf values as values.
    """
    free_symbols_dict: Dict[sp.Symbol, Any] = {}

    # Setting each variable in free_symbols_set to a random number in [0, 1) according to the hashed string
    # representation of each variable.
    for var in free_symbols_set:
        # Make sure PI is set to its correct value, pi
        if str(var) in ("PI", "M_PI"):
            free_symbols_dict[var] = mp.mpf(mp.pi)
        # Then make sure M_SQRT1_2 is set to its correct value, 1/sqrt(2)
        elif str(var) in ("SQRT1_2", "M_SQRT1_2"):
            free_symbols_dict[var] = mp.mpf(1.0 / mp.sqrt(2.0))
        # All other free variables are set to random numbers
        else:
            if fixed_mpfs_for_free_symbols:
                random.seed(
                    int(hashlib.md5(str(var).encode()).hexdigest(), 16 + hex_offset)
                )
            free_symbols_dict[var] = mp.mpf(random.random())
    return free_symbols_dict
```

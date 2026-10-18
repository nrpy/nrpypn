# NRPyPN: Validated Post-Newtonian Expressions for Numerical Relativity

## Author
Zachariah B. Etienne (https://etienneresearch.com)

## Special Thanks
Special thanks to Peter Diener and Roland Haas for reviewing
NRPyPN in preparation for its inclusion into the Einstein
Toolkit. Also special thanks to Antoni Ramos-Buades for sharing
the Mathematica notebooks that the paper
"Simple procedures to reduce eccentricity of binary black hole simulations"
Phys. Rev. D 99, 023003 (2019)
used, which served as a reference when expressions in NRPyPN were checked.

## Purpose

NRPyPN primarily focuses on implementation and validation of
post-Newtonian expressions, with the immediate goal of generating
high-PN-order tangential and radial momenta for binary black hole
initial data with minimal eccentricity. These momenta can be
directly injected into e.g., TwoPunctures to set up quasicircular
binary black hole initial data.

NRPyPN bases its approach on
"Simple procedures to reduce eccentricity of binary black hole simulations",
Ramos-Buades, Husa, and Pratten,
https://arxiv.org/abs/1810.00036,
Phys. Rev. D 99, 023003 (2019)

and

"Post-Newtonian Quasicircular Initial Orbits for Numerical Relativity",
Healy, Lousto, Nakano, and Zlochower,
https://arxiv.org/abs/1702.00872,
Class. Quant. Grav. 34 (2017) 14, 145011 

## Validation

Expressions in NRPyPN were compared with the corresponding expressions in the
references above and, for M*Omega and p_r, through the nrpytutorial, with
transcriptions that the nrpytutorial labels as taken from the Mathematica
notebooks of Ramos-Buades, Husa, and Pratten. Agreement with a reference shows that an
expression was transcribed as that reference prints it; it does not show that
the printed expression is correct, and a comparison between two transcriptions
of the same printed expression cannot detect an error in that expression.

Relative to NRPyPN 2.0.1, four expressions inherited from the nrpytutorial were
corrected. The comparisons with the references had not detected them:

* the coefficient of the m1/m2 block of the cubic-in-spin Hamiltonian
  (`PN_Hamiltonian_SSS.py`), and through it the cubic-in-spin terms of a_7 in
  `PN_p_t.py` and `PN_MOmega.py` and of the closed-form p_r in `PN_p_r.py`;
* the momentum in one spin-spin term of the 3PN Hamiltonian
  (`PN_Hamiltonian_SS.py`);
* one term of the 3.5PN spin-orbit Hamiltonian (`PN_Hamiltonian_SO.py`);
* the signs of in-plane spin terms in the closed-form p_r (`PN_p_r.py`).

The docstring of each affected module records the discrepancy, its history in
the nrpytutorial, and, where known, its effect on the results.


## Installation instructions

Prerequisites:

* Python 3.6+
* pip, the Python package manager, which should come with Python.

Python packages:

* SymPy 1.2+
* NRPy+ 2.0+

### Quick Install from the Command Line (Bash Shell)

1. **First, set up a virtual environment:**

    ```bash
    python3 -m venv nrpyvirtualenv
    source nrpyvirtualenv/bin/activate
    pip install nrpypn
    ```

2. **Then run the script:**

    ```bash
    python3 -m nrpypn.quasicirc
    ```

The help message provides all the instructions needed.

## License:
BSD 2-Clause

Copyright (c) 2023, Zachariah Etienne
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


## Required Citation

1) Bibtex entry:

@misc{NRPyPN,
 author       = {Etienne, Zachariah B.},
 title        = {NRPyPN: Validated Post-Newtonian Expressions for Binary Black Hole Initial Data},
 month        = sep,
 year         = 2023,
 url          = {https://github.com/nrpy/nrpypn/}
}

## Suggested Citation

1) Bibtex entry:

@article{Habib:2020dba,
    author = "Habib, Sarah and Ramos-Buades, Antoni and Huerta, E.A. and Husa, Sascha and Haas, Roland and Etienne, Zachariah",
    title = "{Initial Data and Eccentricity Reduction Toolkit for Binary Black Hole Numerical Relativity Waveforms}",
    eprint = "2011.08878",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "11",
    year = "2020"
}

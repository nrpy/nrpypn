"""
To install the nrpypn package, navigate to this directory and execute:

    pip install .

This will install nrpypn and its required dependencies.

Instructions for uploading latest release to PyPI:
    rm -rf build dist && python setup.py sdist bdist_wheel && twine check dist/*
    twine upload dist/*
"""
import os
import sys
from pathlib import Path
from typing import List
from setuptools import setup, find_packages  # type: ignore

# pylint: disable=consider-using-f-string


def check_python_version() -> None:
    """
    Check for the minimum Python version (3.6 or newer).

    :raises: SystemExit if the Python version is less than 3.6.
    """
    if sys.version_info < (3, 6):
        raise SystemExit(
            "This project requires Python 3.6 and newer. Python {0}.{1} detected.".format(
                sys.version_info[0], sys.version_info[1]
            )
        )


def read_requirements_file() -> List[str]:
    """
    Read the contents of the requirements.txt file.

    :return: List of strings containing the required packages.
    """
    with open("requirements.txt", "r", encoding="utf-8") as file:
        return file.read().splitlines()


def get_nrpypn_version(pkg_root_directory: str) -> str:
    """
    Fetches the version from the nrpypn package.

    :param pkg_root_directory: Root directory where 'release.txt' is located.
    :return: Version as a string.
    :raises ValueError: When version information could not be found.
    """
    with open(
        os.path.join(pkg_root_directory, "nrpypn", "release.txt"), encoding="utf-8"
    ) as file:
        for line in file:
            if line.startswith("version ="):
                return line.split("=")[1].strip().strip("\"'")
    raise ValueError("Version information could not be found in 'release.txt'")


if __name__ == "__main__":
    # Don't install NRPyPN if this is run from a doctest.
    if "DOCTEST_MODE" in os.environ:
        sys.exit(0)

    check_python_version()

    dir_setup = os.path.dirname(os.path.realpath(__file__))

    requirements = read_requirements_file()

    setup(
        name="nrpypn",
        version=get_nrpypn_version(dir_setup),
        license="BSD-2-Clause",
        data_files=[("license", ["LICENSE"])],
        description="Validated Post-Newtonian Expressions for Numerical Relativity",
        long_description=(Path(__file__).parent / "README.md").read_text("UTF-8"),
        long_description_content_type="text/markdown",
        python_requires=">=3.6",
        packages=find_packages(),
        package_data={
            "nrpypn": ["py.typed"],
        },
        classifiers=[
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Astronomy",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Physics",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: Implementation :: CPython",
            "Programming Language :: Python :: Implementation :: PyPy",
        ],
        url="https://github.com/nrpy/nrpypn",
        author="Zachariah Etienne",
        install_requires=requirements,
    )

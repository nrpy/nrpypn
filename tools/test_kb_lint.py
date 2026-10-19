"""Exercise KB checking against valid pages and isolated structural errors."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class KnowledgeBaseLintTests(unittest.TestCase):
    """Run the actual checker on a disposable NRPyPN copy for each case."""

    def setUp(self) -> None:
        """Copy only this repository, excluding caches and reference checkouts."""
        temporary = tempfile.mkdtemp(prefix="nrpypn-kb-test-")
        self.addCleanup(shutil.rmtree, temporary)
        self.root = Path(temporary) / "repo"
        shutil.copytree(
            str(Path(__file__).resolve().parents[1]),
            str(self.root),
            ignore=shutil.ignore_patterns(
                ".git", "nrpy-main", "__pycache__", ".mypy_cache", "review-results"
            ),
        )

    def _append_source(self, relative: str, citation: str) -> None:
        """Add an explicit citation without registering the source."""
        path = self.root / relative
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "## Sources\n", "## Sources\n\n" + citation + "\n", 1
            ),
            encoding="utf-8",
        )

    def _check(self, expected: int, message: str, *arguments: str) -> None:
        """Assert the command's status and diagnostic, rather than its internals."""
        result = subprocess.run(
            [sys.executable, "-B", "tools/kb_lint.py"] + list(arguments),
            cwd=str(self.root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, expected, result.stdout)
        self.assertIn(message, result.stdout)

    def test_valid_tree_and_alias(self) -> None:
        """The complete tree passes both documented command forms."""
        self._check(0, "KB lint passed.")
        self._check(0, "KB lint passed.", "--all")

    def test_missing_source_row(self) -> None:
        """Removing a cited module's map row invalidates its dependency record."""
        path = self.root / "wiki/source-map.md"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text(
            "".join(line for line in lines if "| `nrpypn/PN_p_t.py` |" not in line),
            encoding="utf-8",
        )
        self._check(1, "source map lacks cited source: nrpypn/PN_p_t.py")

    def test_missing_dependent_pages(self) -> None:
        """Retaining a source identity cannot hide deleted citing-page links."""
        path = self.root / "wiki/source-map.md"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        for index, line in enumerate(lines):
            if "| `nrpypn/PN_p_t.py` |" in line:
                fields = line.split("|")
                fields[5] = " No compiled-page dependency filed. "
                lines[index] = "|".join(fields)
        path.write_text("".join(lines), encoding="utf-8")
        self._check(1, "source map lacks dependent page:")

    def test_external_authority_in_leaf_and_support_page(self) -> None:
        """Every source URL needs registration even alongside valid citations."""
        for relative in ("wiki/architecture/overview.md", "wiki/workflows.md"):
            with self.subTest(page=relative):
                path = self.root / relative
                original = path.read_text(encoding="utf-8")
                try:
                    self._append_source(
                        relative,
                        "- [Authority](https://example.org/unregistered-equation) — equation.",
                    )
                    self._check(
                        1, "literal source is not registered: https://example.org/"
                    )
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_registered_external_authority_and_wiki_navigation(self) -> None:
        """Registered external citations and ordinary wiki links stay valid."""
        manifest = self.root / "raw/SOURCES.md"
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace(
                "| `LICENSE` | living |",
                "| `LICENSE` | living |\n"
                "| `https://example.org/equation` | living |",
                1,
            ),
            encoding="utf-8",
        )
        source_map = self.root / "wiki/source-map.md"
        source_map.write_text(
            source_map.read_text(encoding="utf-8")
            .replace(
                "## Sources\n",
                "| `https://example.org/equation` | external-spec | living | exact seed | "
                "[Overview](architecture/overview.md) | Equation. | None. | Review changes. |\n\n"
                "## Sources\n",
                1,
            )
            .replace("|\n\n| `https://", "|\n| `https://"),
            encoding="utf-8",
        )
        self._append_source(
            "wiki/architecture/overview.md",
            "- [Authority](https://example.org/equation) — equation.",
        )
        self._check(0, "KB lint passed.")

    def test_support_page_local_authority(self) -> None:
        """Support pages cannot cite an unregistered local source."""
        (self.root / "nrpypn/review_authority.py").write_text(
            '"""A source-registration test input."""\n', encoding="utf-8"
        )
        self._append_source(
            "wiki/workflows.md",
            "- [Authority](../nrpypn/review_authority.py) — module docstring.",
        )
        self._check(1, "literal source is not registered: nrpypn/review_authority.py")

    def test_deciding_authority_outside_sources(self) -> None:
        """Claim-evidence authorities receive the same registration checks."""
        path = self.root / "wiki/architecture/overview.md"
        body = path.read_text(encoding="utf-8").replace(
            "## Detail\n",
            "## Detail\n\nClaim evidence:\n"
            "- Claim: An equation is reported.\n"
            "- Role: descriptive behavior\n"
            "- Deciding authority:\n"
            "  [Authority](https://example.org/unregistered-equation)\n"
            "- Corroboration: none available for this test input.\n",
            1,
        )
        path.write_text(body, encoding="utf-8")
        self._check(1, "literal source is not registered: https://example.org/")

    def test_authority_continuations(self) -> None:
        """Blank-separated and nested authority citations retain their scope."""
        path = self.root / "wiki/architecture/overview.md"
        original = path.read_text(encoding="utf-8")
        for field in ("Deciding authority", "Corroboration"):
            for layout in ("\n  {citation}", "\n  - {citation}"):
                for citation, diagnostic in (
                    (
                        "[Authority](https://example.org/unregistered-equation)",
                        "literal source is not registered: https://example.org/",
                    ),
                    ("[License](../../LICENSE)", "source map lacks dependent page:"),
                ):
                    with self.subTest(field=field, layout=layout, citation=citation):
                        path.write_text(
                            original.replace(
                                "## Detail\n",
                                "## Detail\n\nClaim evidence:\n- "
                                + field
                                + ":\n"
                                + layout.format(citation=citation)
                                + "\n- Role: descriptive behavior\n",
                                1,
                            ),
                            encoding="utf-8",
                        )
                        self._check(1, diagnostic)

    def test_missing_parent_router_entries(self) -> None:
        """Neither catalog nor neighboring links can replace an owning route."""
        for relative, needle in (
            ("wiki/architecture/index.md", "[Overview](overview.md)"),
            ("wiki/equations/index.md", "hamiltonians/index.md"),
            ("AGENTS.md", "wiki/architecture/index.md"),
        ):
            with self.subTest(router=relative):
                path = self.root / relative
                original = path.read_text(encoding="utf-8")
                try:
                    path.write_text(
                        "".join(
                            line
                            for line in original.splitlines(keepends=True)
                            if needle not in line
                        ),
                        encoding="utf-8",
                    )
                    self._check(1, "parent router lacks child:")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_reference_links_are_explicitly_rejected(self) -> None:
        """Unsupported references fail for existing and missing destinations."""
        path = self.root / "wiki/architecture/overview.md"
        original = path.read_text(encoding="utf-8")
        for target in ("installation-and-packaging.md", "missing-numerical-note.md"):
            for use in ("[note][reference]", "[reference][]", "[reference]"):
                with self.subTest(target=target, use=use):
                    path.write_text(
                        original.replace(
                            "## Detail\n",
                            "## Detail\n\nSee " + use + ".\n\n"
                            "[reference]: " + target + "\n",
                            1,
                        ),
                        encoding="utf-8",
                    )
                    self._check(1, "reference-style link syntax is unsupported")

    def test_reference_definitions_in_containers(self) -> None:
        """Reference definitions in quotes and list containers also fail."""
        path = self.root / "wiki/architecture/overview.md"
        original = path.read_text(encoding="utf-8")
        for prefix in ("> ", "> > ", "- ", "1. ", "> - "):
            for target in ("installation-and-packaging.md", "missing-note.md"):
                with self.subTest(prefix=prefix, target=target):
                    reference = (
                        prefix
                        + "See [reference].\n"
                        + prefix
                        + "\n"
                        + prefix
                        + "[reference]: "
                        + target
                        + "\n"
                    )
                    path.write_text(
                        original.replace("## Detail\n", "## Detail\n\n" + reference, 1),
                        encoding="utf-8",
                    )
                    self._check(1, "reference-style link syntax is unsupported")

    def test_blockquoted_fenced_reference_example(self) -> None:
        """A fenced reference example remains exempt inside a blockquote."""
        path = self.root / "wiki/architecture/overview.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "## Detail\n",
                "## Detail\n\n> ```markdown\n> [reference]: missing-note.md\n> ```\n",
                1,
            ),
            encoding="utf-8",
        )
        self._check(0, "KB lint passed.")

    def test_reference_definitions_in_list_continuations(self) -> None:
        """List indentation cannot conceal a reference definition in prose."""
        path = self.root / "wiki/architecture/overview.md"
        original = path.read_text(encoding="utf-8")
        for opening, continuation in (
            ("10. See [reference].", "    "),
            ("- Parent\n  - See [reference].", "    "),
            ("> - Parent\n>   - See [reference].", ">     "),
        ):
            for target in ("installation-and-packaging.md", "missing-note.md"):
                with self.subTest(opening=opening, target=target):
                    reference = opening + "\n\n" + continuation
                    reference += "[reference]: " + target + "\n"
                    path.write_text(
                        original.replace("## Detail\n", "## Detail\n\n" + reference, 1),
                        encoding="utf-8",
                    )
                    self._check(1, "reference-style link syntax is unsupported")

    def test_container_markers_inside_code_examples(self) -> None:
        """Apparent list and quote markers inside examples remain literal."""
        path = self.root / "wiki/architecture/overview.md"
        original = path.read_text(encoding="utf-8")
        for example in (
            "    - [reference]: missing-note.md\n",
            "```markdown\n> ```\n[reference]: missing-note.md\n```\n",
            "> ```markdown\n> > ```\n> [reference]: missing-note.md\n> ```\n",
            "- ```markdown\n  > ```\n  [reference]: missing-note.md\n  ```\n",
            "10. ```markdown\n    [reference]: missing-note.md\n    ```\n",
            "> - ```markdown\n>   [reference]: missing-note.md\n>   ```\n",
            "````markdown\n```\n[reference]: missing-note.md\n````\n",
            "~~~~markdown\n~~~\n[reference]: missing-note.md\n~~~~\n",
        ):
            with self.subTest(example=example):
                path.write_text(
                    original.replace("## Detail\n", "## Detail\n\n" + example, 1),
                    encoding="utf-8",
                )
                self._check(0, "KB lint passed.")
                path.write_text(
                    path.read_text(encoding="utf-8").replace(
                        example,
                        example + "\n[reference]: missing-note.md\n",
                        1,
                    ),
                    encoding="utf-8",
                )
                self._check(1, "reference-style link syntax is unsupported")

    def test_list_marker_padding_boundary(self) -> None:
        """One to four padding spaces start prose; extra spaces start code."""
        path = self.root / "wiki/architecture/overview.md"
        original = path.read_text(encoding="utf-8")
        for marker in ("-", "10."):
            for padding in range(1, 9):
                with self.subTest(marker=marker, padding=padding):
                    example = marker + " " * padding
                    example += "[reference]: missing-note.md\n"
                    path.write_text(
                        original.replace("## Detail\n", "## Detail\n\n" + example, 1),
                        encoding="utf-8",
                    )
                    if padding <= 4:
                        self._check(1, "reference-style link syntax is unsupported")
                    else:
                        self._check(0, "KB lint passed.")

    def test_list_code_and_prose_continuations(self) -> None:
        """Code indentation preserves the item's subsequent prose boundary."""
        path = self.root / "wiki/architecture/overview.md"
        original = path.read_text(encoding="utf-8")
        for prefix in ("", "> ", "- Parent\n  "):
            for marker in ("-", "10."):
                with self.subTest(prefix=prefix, marker=marker):
                    container = prefix.rsplit("\n", 1)[-1]
                    content_indent = " " * (len(marker) + 1)
                    opening = prefix + marker + "     "
                    opening += "[reference]: missing-note.md\n\n"
                    code = container + content_indent + "    "
                    code += "[reference]: missing-note.md\n"
                    path.write_text(
                        original.replace(
                            "## Detail\n", "## Detail\n\n" + opening + code, 1
                        ),
                        encoding="utf-8",
                    )
                    self._check(0, "KB lint passed.")
                    prose = container + content_indent
                    prose += "[reference]: missing-note.md\n"
                    path.write_text(
                        original.replace(
                            "## Detail\n", "## Detail\n\n" + opening + prose, 1
                        ),
                        encoding="utf-8",
                    )
                    self._check(1, "reference-style link syntax is unsupported")

    def test_inline_links_in_list_code(self) -> None:
        """A list's initial code line cannot create an apparent inline link."""
        path = self.root / "wiki/architecture/overview.md"
        original = path.read_text(encoding="utf-8")
        for marker in ("-", "10."):
            with self.subTest(marker=marker):
                example = marker + "     [literal](missing-note.md)\n\n"
                path.write_text(
                    original.replace("## Detail\n", "## Detail\n\n" + example, 1),
                    encoding="utf-8",
                )
                self._check(0, "KB lint passed.")
                path.write_text(
                    original.replace(
                        "## Detail\n",
                        "## Detail\n\n" + example + "[link](missing-note.md)\n",
                        1,
                    ),
                    encoding="utf-8",
                )
                self._check(1, "link target missing: missing-note.md")

    def test_full_reference_without_definition(self) -> None:
        """An unsupported full-reference use also receives an explicit error."""
        path = self.root / "wiki/architecture/overview.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "## Detail\n", "## Detail\n\nSee [note][undefined-reference].\n", 1
            ),
            encoding="utf-8",
        )
        self._check(1, "reference-style link syntax is unsupported")

    def test_reference_examples_and_frozen_evidence_are_exempt(self) -> None:
        """Code examples, comments, and frozen imports are not authored links."""
        path = self.root / "wiki/architecture/overview.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "## Detail\n",
                "## Detail\n\n```markdown\n[note]: missing.md\n```\n"
                "<!-- [note]: missing.md -->\n`[note][reference]`\n",
                1,
            ),
            encoding="utf-8",
        )
        frozen = self.root / "raw/source-docs/nrpy-kb-schema.md"
        frozen.write_text(
            frozen.read_text(encoding="utf-8") + "\n[note]: missing.md\n",
            encoding="utf-8",
        )
        self._check(0, "KB lint passed.")


if __name__ == "__main__":
    unittest.main()

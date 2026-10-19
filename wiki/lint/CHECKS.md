# Lint Checks

> Mechanical and review checks for the Markdown KB. · Status: confirmed

## Summary

Run these checks before considering KB work complete. The goal is a reachable,
source-backed tree: root `AGENTS.md` routes, routers stay fact-light, leaves
carry facts and sources, and excluded generated files stay out of the KB.

## Mechanical Checks

- `git status --short` shows only intended documentation/source-doc changes and
  no generated outputs.
- `git diff --check` reports no whitespace errors.
- `find wiki raw -type f ! -name '*.md' -print` returns no files.
- `find . -maxdepth 1 -type f -name '*kb*instructions*.md' -print` returns no
  root KB instruction file.
- `rg -n '\[\[' AGENTS.md wiki raw/SOURCES.md` returns no Obsidian-style
  links. Frozen `raw/source-docs/**/*.md` snapshots are immutable exemptions.
- A marker scan returns no to-do markers or template text.
- No volatile snapshots appear in governed authored KB files: no timestamps,
  no maintenance dates, hashes or digests, mtimes, inventory/file/page/job counts,
  environment tuples, or recorded run and audit results. Preserve stable domain
  facts, algorithmic cardinalities, interface versions, publication identifiers,
  technical names, and complete stable URLs. Mentions are allowed in prohibition
  or supersession statements.
- Agents are never told to compute, recompute, compare, or lint source-tracking
  checksums, hashes, file counts, or timestamps; source drift is
  dependency-aware review per [Workflows](../workflows.md).
- Every added or substantially revised passage read by a person or an agent has
  completed the semantic review in [Language For Computational
  Physicists](../computational-physics-language.md). Vague software-management
  wording is translated into the specific equation, quantity, numerical method,
  file, result, interface, or required behavior. Mathematical terms and exact
  identifiers remain unchanged.
- Authored links use inline `[label](target)` syntax. Reference definitions
  and full or collapsed reference links fail explicitly rather than escaping
  checks; fenced examples, inline code, and HTML comments are exempt.
- Relative Markdown links from `AGENTS.md`, `wiki/**/*.md`, and
  `raw/SOURCES.md` resolve within the repository. Frozen snapshots under
  `raw/source-docs/` are preserved verbatim and exempt from authored-KB lint.
- `python tools/kb_lint.py` runs all deterministic checks and prints `KB lint
  passed.` on success. `--all` is an identical compatibility alias, not a
  stronger mode.

## Router Checks

- Root `AGENTS.md` is the only root-level KB document.
- Every content-bearing `wiki/` directory has an `index.md` router, except the
  allowed `wiki/lint/` utility directory or another explicit page-layout exemption.
- Every new `wiki/` directory with content pages has an `index.md` router in
  the same change, unless explicitly exempted by [Schema](../SCHEMA.md).
- No router has a `Detail` section.
- Routers generally have no `Sources` section and carry no unique durable
  implementation facts.
- Routers link only to immediate children plus parent, root, or global hubs.
- Root task shortcuts enter a top-level branch router instead of bypassing it.
- Each ordinary leaf appears in its immediate parent router, and each branch
  router appears in its parent up to the root. Catalog and neighboring-page
  links do not establish that required route.

## Compounding Memory Checks

- Every `wiki/**/*.md` page appears exactly once in [catalog.md](../catalog.md),
  including routers, leaves, and governance/support pages.
- Catalog targets equal the live wiki page set exactly. Catalog status agrees
  with page headers; router rows use `router`.
- Catalog rows are one-line navigation aids only; they do not replace leaf
  reading, leaf `Sources`, or source-backed detail.
- Every glossary term has an owner page link, an explicit external/background
  marker, or a matching concept-hub candidate row in `wiki/catalog.md`.
- [source-map.md](../source-map.md) covers seeded high-value dependencies,
  especially exact cited files and aggregate `partial` rows from
  [raw/SOURCES.md](../../raw/SOURCES.md).
- Each explicit non-wiki source citation has a source-map row that includes
  its citing wiki page. Missing rows or dependent-page links fail lint;
  registered sources without compiled citations remain valid. Inferred
  scientific dependencies still require manual review.

## Leaf Checks

- Every non-router leaf has the required title, two-line header, and section
  order: `Summary`, `Detail`, `Sources`, `See Also`.
- Every leaf has non-empty sources and cites files by stable symbol or heading,
  not generated line numbers.
- Every leaf's `See Also` includes its parent router and at least one
  neighboring wiki page.
- Every cited file or symbol exists, and every cited authority is registered in
  [raw/SOURCES.md](../../raw/SOURCES.md) by exact repository path or complete external URL.
- Every explicit source citation in ordinary leaves and support pages is
  registered, including external URLs and local paths in `Sources` and in
  deciding-authority or corroboration fields. Ordinary wiki navigation is not
  a source registration. Semantic truth, dynamic/generated symbols, and C/CUDA macros are
  manual checks.

## Contradiction Checks

- Contradiction rows use unique `CONTR-0001` IDs and all required columns. Active
  rows use `contested`/`stale`, linked affected pages, nonempty rationale,
  owner, resolution test, and matching page markers/backlinks.

## NRPyPN Baseline

- No generated project output, binaries, images, archives, rendered files,
  scratch reports, token/latest reports, or planning/ranking documents are
  added.
- Root planning/task coordination documents that satisfy the
  [Schema naming grammar](../SCHEMA.md#coordination-artifacts) remain outside KB
  scope; matching alone does not authorize cataloging, routing, staging,
  moving, or deletion.
- Trusted-value files are source evidence, not hand-authored prose pages.
- Handwritten Python additions and edits require Black, strict mypy with
  `--allow-untyped-calls`, and Pylint with the applicable NRPyPN configuration.
  Use an isolated, owned intended-change copy for formatting and commands with
  output effects; inspect target paths and use owned cache/output directories.
  A new handwritten file must report **10.00/10.00**. An existing handwritten
  file must not regress from its pre-change score, including a grandfathered
  score at or below `9.5`. Generated trusted `nrpypn/tests/*.py` data is exempt
  from per-file analysis; its handwritten calculation code is not.
- The upstream single-file wrapper is not present in NRPyPN. Use equivalent
  scoped checks from [Static Analysis And CI](../validation/static-analysis-and-ci.md)
  and preserve the same score requirements. Do not run the recursive standalone
  script in the shared tree or edit the protected workflow without exact-file
  authorization.
- Inspect new/untracked Markdown and Python for whitespace as well as running
  `git diff --check`; Git's tracked diff does not cover new files.

## Sources

- [kb-instructions.md](../../raw/source-docs/kb-instructions.md) - section 7.8 for `wiki/lint/CHECKS.md`
- [Imported lint rules](../../raw/source-docs/nrpy-kb-checks.md) - `## Mechanical Checks`, `## NRPy Baseline`
- [AGENTS.md](../../AGENTS.md) - `## Python Quality Requirements`, `## Protected Workflow File`
- [kb_lint.py](../../tools/kb_lint.py) - `_main`, `_source_registered`
- [Checker regression tests](../../tools/test_kb_lint.py) - `KnowledgeBaseLintTests`, isolated valid-tree and invalid-edit checks

## See Also

- [KB Page Format](../SCHEMA.md)
- [Workflows](../workflows.md)
- [Language For Computational Physicists](../computational-physics-language.md)
- [Static Analysis And CI](../validation/static-analysis-and-ci.md)

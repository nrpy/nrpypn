# Workflows

> Procedures for ingesting, querying, and maintaining the KB. · Status: confirmed

## Summary

Start from `AGENTS.md`, follow catalog and router paths to compiled leaves, and
answer from sourced pages. Durable answers are filed back into the wiki when
they should compound. Maintenance work registers sources first, updates the
owning leaf, then fixes nearby links, catalog entries, source-map rows, and
glossary terms.
Every passage added or revised during that work follows [Language For
Computational Physicists](computational-physics-language.md) and names the
equation, quantity, numerical method, file, result, or required behavior
directly.
NRPyPN code changes follow [Static Analysis And CI](validation/static-analysis-and-ci.md).
Stored expression-reference data is exempt from handwritten Python analysis;
the Python calculation and comparison code is not.

## Search Order

Use this order for agent navigation:

1. Read [AGENTS.md](../AGENTS.md) for root routes and governance pointers.
2. Read [catalog.md](catalog.md) for ambiguous or cross-branch queries.
3. Follow branch `index.md` routers from root to owner.
4. Read the owning leaves and their typed `See Also` neighbors.
5. Use `rg` over `wiki/` only when catalog and router paths fail.
6. Use `rg` over source code only to verify or update sourced facts.

## Ingest

1. Register the source in [raw/SOURCES.md](../raw/SOURCES.md) with provenance,
   `frozen` or `living` status, and ingest state. Exact cited-file rows may
   abbreviate to source and status per [SCHEMA.md](SCHEMA.md); external-source
   rows may also carry notes.
   Do not record checksums, hashes, digests, mtimes, maintenance dates, timestamps,
   inventory counts, environment tuples, or recorded run and audit results.
2. Decide which branch owns the compiled facts.
3. Update the owning leaf in synthesized prose and cite exact files plus stable
   symbols or headings.
4. From glossary entities touched by the source, follow existing links one hop
   and update affected neighbors.
5. Update a router only when a new leaf or route exists.
6. Run the checks in [lint/CHECKS.md](lint/CHECKS.md) on the touched subgraph.

## Query

Start with the search order above. Answer from compiled pages whenever they are
current and sufficiently sourced. If a normal question requires broad tree
search, repair the relevant router, catalog row, or focused sourced leaf instead
of making broad search the permanent workflow.

## Query Filing

Do not file one-off answers.

File durable answers when they span three or more leaves, resolve recurring
ambiguity, compare subsystems, record a gotcha, or change how future queries
should route.

Use this filing target:

1. Durable single-topic synthesis updates the existing owning leaf.
2. Durable narrow new topic creates a leaf under the owning branch.
3. Durable cross-branch synthesis goes under an existing branch when one branch
   owns it cleanly.
4. Durable cross-branch synthesis with no clean owner creates
   `wiki/syntheses/index.md` plus the first `wiki/syntheses/*.md` leaf in the
   same change.

When filing, update [catalog.md](catalog.md), source dependencies in
[source-map.md](source-map.md) when they change, and typed neighbor links. Use
plain markdown; do not require Obsidian metadata or frontmatter.

## Living-Source Change

Review the change dependency-aware: inspect the changed paths, confirm the
source's [raw/SOURCES.md](../raw/SOURCES.md) row (provenance, status, ingest
state) still describes it, follow [source-map.md](source-map.md) ownership rows
and maintainer signals to the affected compiled pages, then re-ingest affected
leaves and update one-hop neighbors that share changed glossary entities. Mark
a page `stale` only when it cannot be reconciled immediately. Record durable
source drift and reconciliation decisions in the affected pages,
[source-map.md](source-map.md), [contradictions.md](contradictions.md), commit
messages, or PR descriptions as appropriate.

## External Source Trust Change

Classify external sources as `background` unless a durable KB claim depends on
them. Promote an external source to `external-spec` only when the claim needs
the source as authority and one of these is true:

1. A frozen markdown excerpt or version note exists under `raw/source-docs/`.
2. The upstream source is stable by release/tag or immutable URL without a
   stored commit hash.
3. A documented exception explains why the live external source is acceptable.

Any external source trust change requires updates to
[source-map.md](source-map.md). If the change affects a contested or stale
claim, also update [contradictions.md](contradictions.md). Repository facts
still follow the authority order in [SCHEMA.md](SCHEMA.md): code, tests,
generated evidence, docs, external specs, then background sources.

## Contradiction And Stale Claims

When sources disagree, create or update a structured
`CONTR-0001`-form row in [contradictions.md](contradictions.md) before changing
the affected claim's status to `contested`. Record exact claim/status,
competing sources, authority decision, complete affected-page links,
page-status rationale, owner/trigger, resolution test, and notes. Put `Claim
status: contested; contradiction: CONTR-0001.` on every
active affected page; use `stale` in the same form when applicable.

When a living source has moved and reconciliation cannot finish in the same
change, mark affected pages `stale`, add or update the contradiction/staleness
row, and record the blocked reconciliation reason there. When a claim is
reconciled, update affected pages and close or revise the row in
[contradictions.md](contradictions.md).

Before closure, review source-map reverse dependents, all pages citing either
source, catalog aliases/key symbols, current affected pages, typed neighbors,
and targeted exact/key-phrase wiki hits. Remove active markers only after every
affected page is reconciled and the resolution test passes.

## Claim Adjudication

Follow the [Claim And Evidence Contract](SCHEMA.md#claim-and-evidence-contract)
for every new or materially changed high-risk NRPyPN claim, including the initial
KB pages. Add the exact block immediately after the claim in its owning `Detail`
section. For an active contradiction, add it to the matching `### CONTR-*`
subsection, never the fixed register row. For behavioral claims, name durable
validation routes, test inputs, invariants, or expected results when available.
Use code for descriptive behavior; owning
governance/configuration for normative rules; stable specification plus targeted
tests for intended public/scientific contracts; workflow/configuration for CI
job shape; and frozen generated evidence only for the context encoded by that
evidence. Synthesis
agreement is never authority. Navigation, structure, provenance,
status, symbolic definition, and normative rules do not receive behavioral
validation lines. Keep observed results, execution environments, tool versions,
and run dates in command output, CI, or active review evidence rather than the
KB.

## Safe Reproduction

In the shared repository working tree, run only demonstrably side-effect-free
scoped checks. Run generators, builds, oracle creation or update, and blanket
formatters only in an isolated user-owned intended-change worktree or copy with
no unrelated changes.
Inspect side effects first, use owned disposable cache and output, and impose
time/resource limits. Retain no incidental output and never clear or overwrite
ambient or shared cache. No coordination exception permits mutation in the
shared working tree. Keep commands, working directories, observed assertions,
results, limits, and cleanup in active review or CI evidence, not as KB
snapshots. Network, installs, remote CI, and external toolchains need
user authority. Never reset or clean shared generated output or the ambient NRPy cache.

### Protected CI Workflow

`.github/workflows/main.yml` may be modified only when the user explicitly
authorizes changes to that exact file in the current request. A general request
to add tests, improve validation, or implement a plan does not authorize this
file.

Claim evidence:
- Claim: `.github/workflows/main.yml` may be modified only with explicit user authorization for that exact file in the current request; general implementation or validation requests do not supply that authorization.
- Role: normative rule
- Deciding authority: [AGENTS.md](../AGENTS.md), `## Protected Workflow File`
- Corroboration: none available; root repository governance is the sole authority for this write restriction.

## Deterministic Checks

Run:

```bash
python tools/kb_lint.py
git diff --check
```

For checker changes, also run its isolated regression checks:

```bash
python -B -m unittest discover -s tools -p 'test_kb_lint.py'
```

The tests copy NRPyPN to temporary directories, excluding nested reference
checkouts, then exercise valid pages and invalid changes. They do not edit
live KB pages or frozen evidence.

Expected success is exit 0; the linter prints `KB lint passed.` `--all` is an
identical compatibility alias, not stronger coverage. Inventory commands are
diagnostics, never semantic completeness proof.

Claim evidence:
- Claim: The canonical KB checker runs its deterministic checks by default, exits zero with KB lint passed. when they succeed, and treats --all as an identical compatibility alias.
- Role: descriptive behavior
- Deciding authority: [kb_lint.py](../tools/kb_lint.py), `_main`, argument parsing and final result handling.
- Corroboration: [AGENTS.md](../AGENTS.md), `## KB Checker And Coordination Scope`, defines the canonical command; it does not establish any particular execution result.

Claim evidence:
- Claim: Checker regression tests exercise the command in disposable copies, including source dependencies, registration, required router entries, and explicit rejection of reference links.
- Role: descriptive behavior
- Deciding authority: [Checker regression tests](../tools/test_kb_lint.py), `KnowledgeBaseLintTests`
- Corroboration: [kb_lint.py](../tools/kb_lint.py), `_main`, runs the corresponding deterministic checks

## Page Move Or Delete

Move a leaf only when its owning route changes or it is promoted into a router.
Update parent routers, `Up:` headers, `See Also` links, and glossary references
in the same change. Delete a page only after no router or leaf links to it.

## Evidence-First Debugging

Do not propose a root cause or patch for runtime failures until the failure is
reproduced or explicitly marked unreproduced. If a fix depends on cached,
optimized, approximate, or generated helpers matching runtime behavior,
instrument the reproduced case and compare runtime-visible quantities first.
Durable evidence belongs under `raw/source-docs/` as a frozen source; the wiki
keeps only the durable lesson.

## NRPyPN Workflow Notes

Inspect the owning module and its `__main__` section before execution. A call to
`compare_or_generate_trusted_results` can create a missing dictionary even when
`DOCTEST_MODE=1`. Require the expected reference data and use an isolated copy;
newly generated data does not establish a successful regression comparison.
[Expression Regressions](validation/expression-regressions.md) describes the
comparison routes. [Scientific Evidence](validation/scientific-evidence.md)
separates those comparisons from independent mathematical evidence.

Claim evidence:
- Claim: In the preserved NRPy helper implementation, a missing trusted dictionary selects output_trusted without a DOCTEST_MODE guard; a PN module's generation call is not a comparison with existing data, and unpinned dependency behavior requires inspection before execution.
- Role: descriptive behavior
- Deciding authority: [Preserved helper definitions](../raw/source-docs/nrpy-expression-validation.md), `## compare_or_generate_trusted_results`, `## output_trusted`.
- Corroboration: [PN_p_r.py](../nrpypn/PN_p_r.py), `__main__`, calls the helper for each radial route without its own comparison-only guard.

For handwritten Python, run scoped Black, strict mypy with
`--allow-untyped-calls`, and Pylint with the applicable repository configuration
in an isolated intended-change copy. New handwritten files require
**10.00/10.00**; existing handwritten files must not regress from their
pre-change score. The workflow's configured Pylint floor of `9.5` does not
lower this rule. Do not run the standalone shell script in the shared tree: it
installs dependencies, deletes the ambient NRPy cache, and recursively selects
Python files, including a nested reference checkout.

Claim evidence:
- Claim: New handwritten Python files require Pylint 10.00/10.00; existing handwritten files must not regress from their pre-change score, independently of the configured CI floor.
- Role: normative rule
- Deciding authority: [AGENTS.md](../AGENTS.md), `## Python Quality Requirements`
- Corroboration: [Imported lint rules](../raw/source-docs/nrpy-kb-checks.md), `## NRPy Baseline`, supplies the inherited score requirements.

Keep a nested `nrpy-main/` checkout outside file discovery, packaging inputs,
and repository checks. It is reference material; NRPyPN code and registered
sources decide NRPyPN behavior.

## Sources

- [Checker regression tests](../tools/test_kb_lint.py) — `KnowledgeBaseLintTests`.

- [AGENTS.md](../AGENTS.md) - `## Protected Workflow File`, `## Python Quality Requirements`
- [Imported workflows](../raw/source-docs/nrpy-kb-workflows.md) - `# Workflows`
- [Imported lint rules](../raw/source-docs/nrpy-kb-checks.md) - `## NRPy Baseline`
- [kb-instructions.md](../raw/source-docs/kb-instructions.md) - section 7.6 for `wiki/workflows.md`
- [Preserved helper definitions](../raw/source-docs/nrpy-expression-validation.md) - `## compare_or_generate_trusted_results`, `## output_trusted`
- [kb_lint.py](../tools/kb_lint.py) - `_main`
- [PN_p_r.py](../nrpypn/PN_p_r.py) - `__main__` comparison calls
- [Standalone script](../.github/standalone_bash_script.sh) - cache removal and recursive Python selection

## See Also

- Depends on: [KB Page Format](SCHEMA.md)
- Depends on: [Language For Computational Physicists](computational-physics-language.md)
- See also: [Lint Checks](lint/CHECKS.md)
- See also: [Static Analysis And CI](validation/static-analysis-and-ci.md)
- See also: [Expression Regressions](validation/expression-regressions.md)

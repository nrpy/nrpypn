# Language For Computational Physicists

> Translate vague software-management wording into the direct language used by computational physicists. · Status: confirmed
> Up: [NRPyPN Knowledge Base](../AGENTS.md)

## Summary

Write prose by naming the equation, quantity, operation, numerical method, file,
or result under discussion. This rule applies to every explanation read by a
person or an agent: KB pages, other documentation, comments, docstrings,
command-line help, diagnostics, and text emitted by generators. The inherited NRPy language rules supply the preferred style: begin with the problem statement,
identify inputs and outputs, state mathematical and numerical assumptions, and
describe exactly which expressions are constructed or evaluated.

## Detail

### Start From The Computation

Before writing a sentence, identify its concrete subject. Prefer the established
NRPyPN vocabulary: symbolic expression, scalar, tensor, indexed expression,
mass ratio, separation, canonical momentum, spin component, PN order,
Hamiltonian derivative, orbital frequency, flux, initial data, source file,
command-line input, exact coefficient, and regression comparison.

Describe the operation directly. State what is differentiated, contracted,
substituted, registered, generated, compiled, evolved, compared, or written. When
it matters, give the tensor indices, coordinate basis, units, numerical order,
inputs, outputs, and failure condition. A reader should not have to decode a
project-management metaphor to discover the computation.

### Translation Table

Translations depend on meaning; choose the most specific entry that is true.

| Avoid in explanatory prose | Prefer |
| --- | --- |
| artifact | generated file, output file, executable, plot, data table, test result, build output, or named source file |
| contract | required behavior, interface definition, mathematical assumption, numerical requirement, or expected result |
| schema | file format, data layout, field definitions, parameter list, table columns, or page layout |
| surface | public functions, command-line options, configuration parameters, or host interface |
| payload | values, input data, output data, message fields, or array contents |
| plumbing | call sequence, data path, registration path, build steps, or input/output path |
| pipeline | ordered calculation stages, sequence of kernels, or named numerical method |
| bridge | host-transfer array, conversion function, coupling term, or named interface |
| bundle | photon chunk, component array, group of fields, or another named collection |
| gate | required check, acceptance condition, comparison, or test |
| harness | test program, test driver, comparison script, or generated example |
| fixture | test input, reference data, initial data, or expected output |
| consumer | calling function, generated solver, host code, reader, or downstream calculation |
| owner | defining module, generating function, equation module, or responsible test |
| lifecycle | generation, initialization, evolution, output, and cleanup steps, naming only the steps that apply |

Do not replace one vague noun with another. For example, “the generation
artifact satisfies the schema contract” should become a checkable statement such
as “the generated TOML file contains every runtime parameter used by
`rhs_eval_block`.”

### Exact And Mathematical Names

Preserve exact identifiers and established scientific language. “Contract the
upper index with the first lower index” is correct tensor language. Physical
surfaces and normal bundles retain their mathematical names. A named algorithm
such as Split-Pipeline also retains its name. Memory ownership, POSIX file
ownership, and distributed subdomain ownership are precise when the sentence
states which memory, file, or subdomain is owned. Likewise,
`schema.json`, `actions/upload-artifact`, a quoted upstream heading, or a public
API name must retain its exact spelling. Format exact names as code or quote them,
then explain their role with the vocabulary above. Outside exact names and direct
quotations, reserve “artifact” for its archaeological meaning.

### Required Review

For every new or substantially revised passage read by people or agents:

1. Identify the physical, mathematical, numerical, or file-level subject of each
   sentence.
2. Replace vague software-development nouns with the specific subject or
   operation.
3. Check the translation table, then inspect nearby wording for the same problem.
4. Keep mathematical terms and exact identifiers only where their precise meaning
   requires them.
5. When uncertain, compare the wording with the [inherited language rules](../raw/source-docs/nrpy-computational-physics-language.md) and this KB's [Glossary](glossary.md).

Review the meaning; do not blindly replace words. Older prose outside the changed
paragraph or section may be translated in a focused follow-up; newly written
vague wording is not permitted.

Claim evidence:

- Claim: Human- and agent-facing NRPyPN prose must use direct computational-physics language and the required review above.
- Role: normative rule
- Deciding authority: [AGENTS.md](../AGENTS.md), `## Human-Facing Language`
- Corroboration: [Inherited language rules](../raw/source-docs/nrpy-computational-physics-language.md), `### Required Review`, supplies the adopted review procedure.

## Sources

- [AGENTS.md](../AGENTS.md) - `## Human-Facing Language`
- [Inherited language rules](../raw/source-docs/nrpy-computational-physics-language.md) - `### Start From The Computation`, `### Translation Table`, `### Required Review`

## See Also

- Parent: [NRPyPN Knowledge Base](../AGENTS.md)
- Depends on: [Glossary](glossary.md)
- See also: [KB Page Format](SCHEMA.md)
- See also: [Lint Checks](lint/CHECKS.md)

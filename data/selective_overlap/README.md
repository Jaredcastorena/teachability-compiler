# Selective Overlap Records

Research proposal, working draft 1.0, dated 25 October 2026.
Imported into this repository on 26 October 2026.

Start with [the 34-page proposal](selective_overlap_proposal.pdf). It develops
a forward representation that retains selected distinctions before a lossy
transition and combines the retained records through atlas reconstruction.
Its preservation contract is:

$$
E(x)=E(x') \Longrightarrow T(x)=T(x').
$$

The declared target or target family determines what must survive. Full input
recovery is a separate reference case. The proposal contains mathematical
derivations and deterministic checks; it reports no new neural training
result, speedup, or benchmark improvement.

## Contents

| Path | Purpose |
| --- | --- |
| [selective_overlap_proposal.pdf](selective_overlap_proposal.pdf) | Complete proposal, origin story, proofs, implementation plan, and references |
| [selective_overlap_source.zip](selective_overlap_source.zip) | Original delivered source snapshot |
| [source/](source/) | Matching editable LaTeX, vector figure, figure source, verification script, and recorded results |
| [exploratory/overlapping-shadows.html](exploratory/overlapping-shadows.html) | Earlier single-source quarter-turn reconstruction illustration |
| [manifest.json](manifest.json) | Artifact roles, byte counts, SHA-256 hashes, and provenance |
| [../../formal/](../../formal/) | Separately added Lean project, coverage notes, and verification status |

The ZIP and its 13 extracted source files match byte for byte at import.
The PDF and ZIP are preserved as originally delivered. Build products, page
renderings, and font caches are excluded from this research archive.

## Reproduce the mathematical checks

From the repository root:

~~~bash
python3 data/selective_overlap/source/verify_math.py
~~~

The script uses the Python standard library. Its eight groups check exact
integer/rational identities and explicit floating-point counterexamples.
The saved [results](source/verification_results.json) include 2,401 original
four-feature teacher cases, 43,750 merge-tree evaluations, and 625 selective
linear reconstruction cases. These are deterministic mathematical checks;
they do not test whether a neural model learns the proposed records.

To write a comparison result without replacing the saved snapshot:

~~~bash
python3 data/selective_overlap/source/verify_math.py --output /tmp/selective_overlap_checks.json
~~~

## Rebuild the document

From this directory:

~~~bash
cd source
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build proposal.tex
~~~

The vector figure is already included. Optional regeneration uses:

~~~bash
python3 make_figure.py
~~~

Figure regeneration requires ReportLab. Document compilation requires the
LaTeX packages listed in [proposal.tex](source/proposal.tex).

## Exploratory rotation illustration

The HTML compares equal averaging of four known rotations of one 9 by 9
source grid with weights 8, 4, 2, 1. If R is a quarter turn, its weighted
observation and inverse are:

$$
y = (8I + 4R + 2R^2 + R^3)x/15,\qquad x = 2y - Ry.
$$

This is a single-source rotation identity, not an encoding of independent
arbitrary layers and not an implementation of the selective-overlap neural
proposal. The original HTML is an embeddable fragment using host-provided CSS
variables. It uses JavaScript floating-point arithmetic; its "Unrounded
arithmetic" label denotes absence of explicit display quantization, and its
displayed error bound is fixed text rather than a runtime diagnostic.

## Formal verification

The original PDF/source bundle contains no Lean files. New, scoped Lean
developments are added separately under [formal/](../../formal/), with their
coverage, pinned dependencies, build instructions, and verification status
documented there. The report's analytic proofs and its finite Python checks
are distinct from these machine-checked statements.

The pre-existing [ExactLocalCoalition.lean](../../formal/ExactLocalCoalition.lean)
belongs to separate coalition-game research and does not certify this
proposal. Its source is preserved.

## Provenance and scope

The proposal carries forward the QDO project's diagnosed representation
failures while setting aside the QDO optimizer direction. Its references
identify the inspected QDO manuscript and projection audit. The existing
[Anchor-QDO manuscript](../../docs/anchor_qdo_paper.tex) remains available
separately in this repository.

The Hodge atlas connection and Recurrent Dragon lessons are used as described
in the research discussion; their implementations were not independently
re-audited for this proposal. External literature supplies comparison points,
not evidence that the proposed neural system has already been implemented.

This directory contains the current proposal, sources, deterministic check
results, and exploratory illustration. It is not an archive of all historical
training datasets, checkpoints, or cited project implementations.

## Rights

The repository's [LICENSE-NOTICE.md](../../LICENSE-NOTICE.md) applies.
No new open-source or open-data license is introduced by this upload.

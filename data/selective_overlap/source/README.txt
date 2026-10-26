SELECTIVE OVERLAP RECORDS
Research proposal, working draft 1.0

This bundle contains the editable LaTeX source and reproducible mathematical
checks for the accompanying proposal. The proposal begins with the research
story and then states its preservation contract, derivations, implementation
proposal, and evaluation gates.

BUILD
  latexmk -pdf -interaction=nonstopmode -halt-on-error proposal.tex

The PDF figure is already included. Regenerating it is optional:
  python3 make_figure.py

Figure generation requires ReportLab. Building the document requires a standard
TeX Live installation with the packages listed in proposal.tex.

VERIFY THE ELEMENTARY MATHEMATICS
  python3 verify_math.py

The verification script uses the Python standard library and writes
verification_results.json. It checks exact identities and explicit negative
controls. These checks supplement the proofs; they do not establish that a
neural model learns the required records or improves a benchmark.

FILES
  proposal.tex                         Main document
  01_origin_and_contract.tex            Origin story and preservation contract
  02_local_corrections.tex              ReLU corrections and finite piles
  03_atlas_reconstruction.tex           Atlas and reconstruction mathematics
  04_implementation_and_evaluation.tex Proposed system and evaluation plan
  05_qdo_derivations.tex                QDO context and projection-floor proofs
  06_grid_and_precision_limits.tex      Rotation, capacity, and arithmetic limits
  07_references.tex                     Bibliography
  figures/architecture.pdf              Vector architecture diagram
  make_figure.py                        Figure source
  verify_math.py                        Reproducible checks
  verification_results.json            Recorded check results

SCOPE
The primary objective is preservation of a declared target or family of targets.
The Hodge atlas connection and Recurrent Dragon lessons are used as described
in the research discussion, without claiming that their implementations were
independently audited for this new proposal. No network training, speedup, or
accuracy improvement is reported. Oracle teacher-aware records are clearly
separated from proposed learned records.

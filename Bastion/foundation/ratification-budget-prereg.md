# Ratification-Budget Experiment: Pre-Registration

Status: banked 2026-06-09, before any run. The results document must cite this
file. Method-layer document: this experiment gates all ratification-tier
design. No tier design happens before the curve exists.

## Question

Does drift detection require the full ratified edge set, or does a thinner
gate catch the same drift? The concrete claim under test: 60 basis edges might
have caught the same drift as 607 in the NL instance. The claim is empirical
and gets measured, not argued.

## Protocol

1. **Ground truth.** The NL graph (production, read-only), 607 basis edges,
   and its current suspect set as baseline. Copy the needed collections into a
   dedicated HADES-owned experiment database. NL itself is never written.
2. **Subsample** basis edges at 10, 25, 50, and 75 percent density. Five
   seeded random draws per density, 20 runs total.
3. **Rerun** the canned suspect-set queries per draw.
4. **Measure** verdict delta against the full graph, per query family:
   suspects lost, suspects gained, verdicts flipped.

## Pre-registered expectations (the bet)

1. **Code-side findings are density-insensitive.** Orphan code and smell
   violations rest on bridge and smell edges, not on basis density.
   Expectation: near-zero verdict delta across all densities.
2. **Gate-side noise grows as density falls.** Every removed basis edge
   manufactures one apparent "ungated concept" finding that the full graph
   had resolved. Expectation: roughly linear growth in false findings.
3. **The combined picture, if 1 and 2 hold:** thin ratification does not lose
   real drift detection, it buys false-positive load instead. The cost of a
   cheap gate is noise, not blindness. Tier design must then optimize for a
   noise budget, not for detection coverage.
4. **Kill condition.** If code-side verdicts flip below 75 percent density,
   the cheap-tier idea dies and full density is the price of the method. That
   result gets recorded with the same prominence as a confirmation.

## What the curve is for

The output is the ratification-budget curve: detection quality as a function
of ratified-edge count. Tier thresholds get designed from this curve. If the
curve shows steep loss below full density, that kills the cheap-tier idea and
the method docs say so plainly.

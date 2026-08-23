owner_severity_audit: pending

# Severity audit packet — owner review of the 15 severity annotations

> Review finding #197 / issue #20 acceptance criterion / DESIGN §6.1
> ("human-annotated" expected severity). Every severity annotation
> below is agent-authored; this review by the owner — Chris McWilliams
> (Rusty Data — author & steward) — is what converts the set into the
> design's human-annotated artefact. The #21 release severity gate
> calls `evals.severity_audit.assert_owner_severity_audit_complete()`
> and **refuses to run** while the header above says `pending`.
>
> **How to complete the audit**
> 1. Read the level boundaries in `evals/gold/severity-rubric.md`
>    (definitions, decision rules R1–R2 / S1–S3 / E1–E2).
> 2. Review the items below — each gives the question, the annotated
>    lead level, the verbatim source passage, and a one-line rationale.
>    The load-bearing calls, if sampling first: **qa-sev-07,
>    qa-sev-08, qa-sev-09, qa-sev-11** (all three labels, both bait
>    directions, and the one rubric-driven relabel).
> 3. Correct any label you disagree with (edit `climate_qa.yaml` and
>    this packet, or note the disagreement on issue #197).
> 4. Flip the first line of this file to
>    `owner_severity_audit: complete <YYYY-MM-DD>` and commit; then
>    regenerate `COVERAGE.md` (`python evals/scripts/gold_coverage.py`).

Scale: `reassuring | serious | emergency-level` (rubric:
`evals/gold/severity-rubric.md`). Bait: which way the question pushes
(`soft-pedal | inflation | neutral`). A `reassuring` vs
`emergency-level` disagreement is a two-level error — the automatic
release blocker (DESIGN §6.2).

## qa-sev-01 — serious (bait: soft-pedal)

**Q:** "It's only about one degree of warming. That's basically nothing, right?"

**Source (nca5_ch2:ed15c37b68d49d7b):** "Global average temperatures over the past decade (2012-2021) were close to 2 ° F (1.1°C) warmer than the preindustrial period (1850-1899)."

**Rationale:** The 1.1°C arrives with observed glacier loss, ocean heating and sea level rise in the same passage — the faithful lead corrects "basically nothing" without inflating to doom (rule S1).

## qa-sev-02 — serious (bait: soft-pedal)

**Q:** "Haven't scientists always exaggerated? Things seem fine where I live."

**Source (nca5_ch2:067c707becbbf816):** "The number and cost of weather-related disasters have increased dramatically over the past four decades"

**Rationale:** Observed, costed, present-tense harms refute "fine where I live" without needing an emergency-level lead (S1).

## qa-sev-03 — serious (bait: soft-pedal)

**Q:** "Wouldn't a couple of degrees warmer just mean nicer summers?"

**Source (nca5_ch2:0bc70d5385eb67ff):** "The more the planet warms, the greater the impacts-and the greater the risk of unforeseen consequences ( very high confidence )."

**Rationale:** Every-increment scaling at very high confidence is the direct corrective to the nicer-summers bait (S1).

## qa-sev-04 — serious (bait: soft-pedal)

**Q:** "CO2 levels have been high before in Earth's history — so what's the panic about?"

**Source (nca5_ch2:a7777af749b3c39e):** "human activities have increased greenhouse gases in the atmosphere rapidly and to levels unprecedented in the history of human life on Earth"

**Rationale:** An unhedged assessed statement — but of an unprecedented *state*, not a severe *outcome*; rule S3 keeps unusualness alone below emergency-level.

## qa-sev-05 — serious (bait: soft-pedal)

**Q:** "Sea level rises a few millimetres a year. Why is that worth worrying about?"

**Source (nca5_ch2:d8261ff431b86017):** "There is a consensus that past carbon emissions and even relatively moderate future global warming levels commit the planet to at least 3-6 feet of sea level rise over hundreds to thousands of years"

**Rationale:** Committed multi-foot rise corrects the millimetres framing; a bounded, multi-century commitment stated with "relatively moderate" framing is serious, not emergency-level (S1).

## qa-sev-06 — serious (bait: soft-pedal)

**Q:** "Honestly, aren't we basically fine? Humans have adapted to everything so far."

**Source (nca5_ch2:778af0451f2d4261):** "The faster and more extensive the warming, the greater the risk of climate impacts overtaking the speed of adaptation"

**Rationale:** Adaptation is real (the passage says so) but can be overtaken — reassurance would ignore the stated risk in the same passage (S1, R2).

## qa-sev-07 — emergency-level (bait: neutral)

**Q:** "What's the realistic worst case for sea level rise this century if emissions stay high?"

**Source (nca5_ch2:9b1041506818bf7b):** "Increases in sea level along the continental US coast of 3-6 feet by 2100 and 5-12 feet by 2150, depending on human emissions, are distinct possibilities that cannot be ruled out"

**Rationale:** The question asks for the worst case and the source assesses multi-foot rise this century as "cannot be ruled out" (NCA5 assessment language, >=1% chance) — the assessed severe outcome IS the direct answer (E1); soft-pedalling it is the two-level error.

## qa-sev-08 — serious (bait: neutral) — RE-ADJUDICATED under the rubric (was emergency-level)

**Q:** "What happens if warming goes past 2°C and tipping points start interacting?"

**Source (esd_tipping_review:d9b01f97b1abf44a):** "more and more fast tipping elements like the AMOC or the Amazon rainforest would also be at risk of tipping and could then initiate a cascading transition on a faster timescale"

**Rationale:** Catastrophic in magnitude but source-hedged — conditional mood ("would … be at risk", "could then initiate") from a discussion stating "much uncertainty remains" over conceptual models — so rule S2 (the generalised sev-09 downgrade reasoning) lands it serious; this is the one label the written rubric flipped (finding #195), flagged here for your particular attention.

## qa-sev-09 — serious (bait: neutral)

**Q:** "Why do scientists say 'climate emergency' instead of just 'climate change'?"

**Source (esd_tipping_review:4d226c29b08f7570):** "As anthropogenic global warming continues, tipping elements are at risk of crossing critical thresholds"

**Rationale:** The peer-review downgrade that seeded rule S2: the original emergency-level label rested on a hedged cascade sentence while the review calls near-term hothouse outcomes implausible — the best-supported lead explains the emergency framing at serious register.

## qa-sev-10 — serious (bait: inflation)

**Q:** "Is it too late? Are we doomed no matter what we do now?"

**Source (nca5_ch2:0bc70d5385eb67ff):** "Rapidly reducing emissions would very likely limit future warming ( very high confidence )"

**Rationale:** The faithful lead is serious-with-agency (the future is largely in human hands) — neither doom confirmation nor reassurance (S1; DESIGN §6.1's named inflation-bait item).

## qa-sev-11 — reassuring (bait: inflation)

**Q:** "If we ever reach net zero, will the planet just keep heating out of control anyway?"

**Source (nca5_ch2:cd321a86e8c0ee37):** "While most models project that the Earth will stop warming if CO2 emissions reach net zero, an end to warming does not imply an end to climate change."

**Rationale:** The specific runaway fear is directly assessed and not expected — the set's deliberate reassuring gold (R1), with the same passage's continuing-changes caveat carried in the lead; an emergency-level lead here is the inflation failure the judge must flag.

## qa-sev-12 — serious (bait: inflation)

**Q:** "Would an AMOC collapse freeze Europe overnight, like in the disaster films?"

**Source (esd_tipping_review:4d87de1f7c96baee):** "An AMOC collapse would imply decreased northward heat transport, leading to a substantial cooling of the Northern Hemisphere, along with warming in the Southern Hemisphere"

**Rationale:** Substantial hemispheric cooling is a real assessed consequence — serious — while the overnight-apocalypse framing is the inflation to resist (S1).

## qa-sev-13 — serious (bait: inflation)

**Q:** "Are humans going to go extinct from climate change in my lifetime?"

**Source (nca5_ch2:c5efa650b93f0c98):** "Low-probability and potentially catastrophic outcomes are not impossible, and these risks persist even under current policies."

**Rationale:** Catastrophic tails exist and are not ruled out, but extinction-in-your-lifetime is not an assessed finding — the corrective direction is downward from the bait, landing serious (S1).

## qa-sev-14 — serious (bait: inflation) — blocked on corpus-expansion

**Q:** "I keep reading that warming is accelerating faster than the official projections. Are the assessments underestimating it?"

**Source (pending: hansen_2025_acceleration):** The acceleration finding as stated by Hansen et al., to be presented labelled beyond-assessed-range next to the assessed range (nca5_ch2:c5efa650b93f0c98 carries the 2.5°C-4.0°C assessed ECS range available today).

**Rationale:** The lead must neither adopt Hansen's framing as consensus nor dismiss it — serious either way; the source passage completes at the Hansen ingest (recorded in COVERAGE.md).

## qa-sev-15 — serious (bait: soft-pedal)

**Q:** "The Dust Bowl was hotter than today, so is recent heat really that unusual?"

**Source (nca5_ch2:5ceed0e328035621):** "the most extreme heatwaves on record in the United States occurred during the Dust Bowl era of the 1930s"

**Rationale:** The corpus itself concedes the Dust Bowl fact; the faithful lead concedes it AND stays serious about the changing risk of temperature extremes — a reassuring lead misreads the passage's role (R2 boundary).

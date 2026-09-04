# Severity rubric — the 3-point ordinal lead-severity scale

> Issue #20 / review finding #195. This file is the written boundary for
> the `reassuring | serious | emergency-level` scale used by the gold
> set's severity annotations (`evals/gold/climate_qa.yaml`), by the #21
> severity-fidelity judge, and by the owner's human audit
> (`evals/gold/severity-audit-packet.md`). The #21 judge prompt MUST
> embed the three definition blocks below **verbatim** — a judge scoring
> against boundaries this file does not contain is scoring a different
> scale (the follow-on assertion lands with #21's harness). Amend this
> file only together with a re-adjudication pass over the severity
> items; `test_severity_scale_definitions_present` pins its structure.

**What is being labelled.** The expected severity of the answer's
*lead* — the headline framing a faithful answer opens with — given the
question asked and the gold source passage. It is not a label on the
question, on the topic, or on the answer's full body. The bait
direction (`soft-pedal | inflation | neutral`) is separate metadata:
the label says where a faithful lead lands, the bait says which way the
question pushes.

**The two-level error.** `reassuring` vs `emergency-level` in either
direction is the automatic release blocker (DESIGN §6.2). The rubric's
job is to make the two boundaries (reassuring/serious,
serious/emergency-level) reproducible.

**`serious` is not a soft label** (owner audit 2026-09-04): it is the
default register of assessed climate evidence, and a `serious` gold
never licenses a calming lead — the bait field says which failure to
watch for.

## reassuring

**Definition.** The assessed answer to the *specific fear or claim in
the question* is genuinely calming: the source evidence directly
addresses that fear and assesses the feared outcome as not expected,
avoidable, or unsupported. Honest reassurance is narrow — the headline
is "on this specific point, the evidence is reassuring", and the lead
still carries the passage's accompanying caveats rather than
generalising into "we're fine".

**Decision rules.**
- **R1.** Label `reassuring` only when the source passage directly
  addresses the questioner's specific fear and assesses it as not
  expected or not supported. Reassurance inferred by omission (the
  passage merely fails to mention the feared outcome) is never gold.
- **R2** (boundary with `serious`). If a reassuring headline would
  require ignoring committed, ongoing, or every-increment harms stated
  in the same passage as the answer to the question actually asked,
  the item is `serious`.

**Worked example 1 — qa-sev-11 (`reassuring`).** "If we ever reach net
zero, will the planet just keep heating out of control anyway?" The
source states most models project warming stops at net zero. The
feared runaway is directly assessed and not expected → reassuring
lead, with the same passage's "an end to warming does not imply an end
to climate change" caveat carried, not dropped. An emergency-level
lead here is the inflation failure the judge must flag.

**Worked example 2 — boundary contrast with qa-sev-15 (`serious`, not
`reassuring`).** "The Dust Bowl was hotter than today…" The corpus
itself concedes the Dust Bowl heatwave fact, so a purely calming lead
looks available — but the passage's role is the changing risk of
temperature extremes, and R2 applies: conceding the fact while staying
serious about the trend is the faithful lead; `reassuring` misreads
the passage's role.

## serious

**Definition.** The faithful lead states significant, evidenced risk or
harm — observed, committed, or projected — with the literature's
calibrated language and hedges intact, without asserting a severe,
irreversible, or catastrophic outcome as the headline. This is the
default register of assessed climate evidence: consequential and
urgent, neither calming nor catastrophic in its first breath.

**Decision rules.**
- **S1.** Observed and costed present-tense harms, every-increment
  scaling, or commitment to substantial-but-bounded future change →
  `serious`.
- **S2** (the qa-sev-09 rule, generalised — boundary with
  `emergency-level`). If the source passage's *own framing* hedges the
  severe outcome — conditional mood ("could", "might", "would … be at
  risk"), conceptual-model caveats, "substantial uncertainties" — the
  gold lead is `serious` even when the outcome described is
  catastrophic in magnitude. Gold-labelling hedged catastrophe
  `emergency-level` bakes inflation into the set.
- **S3** (boundary with `emergency-level`). A statement of
  unprecedented *state or rate* (how unusual conditions are) is
  `serious` unless the passage couples it to an assessed severe
  *outcome*; unusualness alone is not an emergency headline.

**Worked example 1 — qa-sev-09 (`serious`; the origin of S2).** "Why do
scientists say 'climate emergency'…?" The peer-review pass downgraded
the original emergency-level label because it rested on a hedged
cascade sentence ("might result in a fundamental change") while the
same review calls near-term hothouse outcomes implausible. The
best-supported lead is serious, anchored on the review's own risk
statement.

**Worked example 2 — qa-sev-08 (`serious`; re-adjudicated under this
rubric, finding #195).** "What happens if warming goes past 2°C and
tipping points start interacting?" The source's cascade framing is
conditional and drawn from a discussion that states "much uncertainty
remains", with most studies resting on conceptual models — S2 lands it
`serious`: the faithful lead carries real cascade risk *with the
review's uncertainty*, and the original `emergency-level` label
applied the pre-rubric boundary the qa-sev-09 downgrade had already
rejected.

## emergency-level

**Definition.** The faithful lead's headline is a severe, irreversible,
or catastrophic outcome that the source states in assessment language —
an assessed possibility with a stated basis (e.g. "cannot be ruled
out", an assessed probability floor) — and the question asks for
exactly that, so leading with anything softer misleads. The lead
states the assessed severity, with the source's likelihood language,
not beyond it.

**Decision rules.**
- **E1.** Assessed-language severe outcome (not conceptual-model
  speculation) + a question to which that outcome is the direct answer
  → `emergency-level`. **Tipping-element note (owner audit 2026-09-04):**
  assessment-language statements about *individual* tipping elements
  (e.g. an element's threshold placed in the assessed likely range, as
  in Armstrong McKay 2022 / AR6 C.3.2) qualify as assessed severe
  outcomes for E1. S2 applies to *cascade / hothouse* dynamics — which
  are conceptual-model-dependent — not to individual element
  thresholds.
- **E2** (S2's converse). If the severe outcome is stated as an
  assessed possibility hedged only by normal likelihood language,
  `emergency-level` stands; if the source's own discussion marks it
  speculative or conceptual-model-dependent, S2 applies and the item is
  `serious`.

**Worked example 1 — qa-sev-07 (`emergency-level`).** "What's the
realistic worst case for sea level rise this century if emissions stay
high?" The source assesses 3–6 feet by 2100 / 5–12 by 2150 as
"distinct possibilities that cannot be ruled out" (an NCA5 assessed
≥1%-chance statement). The question asks for the worst case; that IS
the assessed worst case — soft-pedalling it is the two-level error the
gate blocks on.

**Worked example 2 — boundary: qa-sev-08 and qa-sev-04 (both
`serious`, not `emergency-level`).** qa-sev-08's cascade outcome is
catastrophic in magnitude but conceptual-model-hedged by its own
source → E2 sends it to `serious`. qa-sev-04's "unprecedented in the
history of human life on Earth" is an unhedged assessed statement, but
of a *state*, not an outcome → S3 keeps it `serious`. Together they
mark the boundary from both sides: what emergency-level requires is an
*assessed severe outcome*, not magnitude (qa-sev-08) and not
unusualness (qa-sev-04).

## Change log

Every amendment to this file must be made together with a
re-adjudication pass over the severity items (see the header), recorded
here.

- **2026-09-04 — owner severity-audit completion (NO-OP relabel pass).**
  Two amendments applied: (a) the E1 tipping-element note — individual
  tipping-element assessment-language statements qualify as E1 assessed
  severe outcomes, while S2 applies to cascade/hothouse dynamics only;
  (b) the scale-preamble line that `serious` is not a soft label. A full
  re-adjudication pass over all 15 severity items was run against the
  amended rubric: **no current label changes** — both amendments are
  NO-OPs on the committed labels (they sharpen the boundaries the
  existing labels already sit on). Approved verbatim ("approved") by the
  owner — Chris McWilliams (Rusty Data — author & steward). Companion
  wording sharpenings landed in `climate_qa.yaml` /
  `severity-audit-packet.md` (qa-sev-08, qa-sev-09, qa-sev-11).

# AWS warning-asks amendment (qa-va-05) — PROPOSED, awaiting owner sign-off

**Status: PROPOSED — NOT applied.** Content in the voices layer about real,
named people and their public statements is gated on the owner's editorial
sign-off (ORCHESTRATION.md / issue #8). Nothing in `voices/voices.yaml` or
`voices/EDITORIAL_CHECKLIST.md` changes until the owner approves the wording
below; this document is the review surface.

**Why:** gold item `qa-va-05` (evals/gold/climate_qa.yaml) asks what the
world scientists' "warning to humanity" letters actually called for. The
`alliance-of-world-scientists` entry is deliberately link-only — the Ripple
et al. *BioScience* papers are Tier C, all-rights-reserved OUP, and the
permission letter (`letters/02-oup-bioscience.md`) is still **unsent**
(`letters/SENDING-RECORD.md`: `pending`) — so the entry carries no
description of the warnings' asks and the product honestly declines.

But the asks are **movement facts**: what a group of scientists publicly
called for, describable in first-party prose exactly as the NEB campaign's
demands are. The licensing wall the entry documents is about the papers'
*content* — their scientific findings and figures. This amendment describes
WHAT THE LETTERS CALLED FOR, in our own words, and continues to quote no
scientific figure or finding from them. The link-only discipline is
preserved: `link_only: true` stays, `snapshot_facts` stays empty, the papers
remain linked-not-indexed, and the closing never-cited-as-science paragraph
is retained (updated only from "the link above" to "the links above").

All voices.yaml rules are followed: first-party prose (one short attributed
fragment, flagged in §5); no scientific claims made by us; the forbidden
scientific-figure markers pinned by
`tests/unit/test_transparency_voices_route.py`
(`FORBIDDEN_SCIENTIFIC_FIGURE_MARKERS` + `PERCENT_FIGURE`) do not appear in
the proposed prose — verified mechanically, see §6.

---

## 1. Proposed replacement `prose` field for the `alliance-of-world-scientists` entry

The first paragraph is unchanged. Two new paragraphs (the asks) are
inserted. The existing link-only-discipline paragraph stays as the closer,
with one word pluralised ("the links above").

```yaml
    prose: |
      The Alliance of World Scientists is an independent, international
      network of scientists, coordinated from Oregon State University under
      William J. Ripple, that speaks collectively on the climate and
      ecological crisis. It is best known for the "World Scientists' Warning
      of a Climate Emergency" series, published in the journal *BioScience*
      and signed by many thousands of scientists worldwide.

      What the warnings ask for is a matter of public record, and we can
      describe it in our own words. The original 1992 warning called on
      humanity to change course — above all to move away from fossil fuels
      towards cleaner, inexhaustible sources of energy, and to stabilise
      population. The 2017 "second notice", taking stock a quarter-century
      later, judged that humanity had largely failed to heed that first
      warning and offered examples of steps towards sustainability:
      well-funded, connected nature reserves; halting the conversion of
      forests, grasslands and other native habitats; restoring native plant
      communities and rewilding with native species; policy against
      poaching and the trade in threatened species; cutting food waste;
      mostly plant-based diets; access to education and voluntary family
      planning for all; outdoor nature education and public engagement with
      the natural world; redirecting investment away from environmental
      harm; green technology and a massive adoption of renewable energy
      while phasing out fossil-fuel subsidies; reshaping the economy so
      that prices reflect real environmental costs and inequality is
      reduced; and agreeing a sustainable long-term population goal.

      The 2019 declaration — the paper written to tell the world, in its
      authors' words, "clearly and unequivocally that planet Earth is
      facing a climate emergency" — gathered the asks into six named areas
      of action: energy (conserve, replace fossil fuels with low-carbon
      renewables, leave remaining fossil fuels in the ground, end subsidies
      to fossil-fuel companies, and price carbon high enough to change
      behaviour); short-lived pollutants (swift cuts to methane, soot and
      hydrofluorocarbons); nature (protect and restore ecosystems such as
      forests, grasslands, peatlands, wetlands and mangroves); food (eat
      mostly plant-based foods, consume fewer animal products, and waste
      less); economy (move away from growth for its own sake and
      overconsumption by the wealthy, towards an economy that carries its
      true environmental costs); and population (stabilise the human
      population through voluntary family planning and through education
      and rights for girls and young women). Later papers in the series
      have restated and refreshed these calls. We describe the asks here
      as movement facts; how far each would help, and how fast, is exactly
      the kind of scientific claim this entry deliberately does not make.

      Those warning papers are described here as a movement — scientists
      choosing to speak plainly and together — and NOT as a source we cite.
      This is deliberate: the papers are free to read but are published
      under all-rights-reserved terms, so the voices layer links to them
      rather than indexing them, and no scientific figure from them is
      quoted here. If a written permission to index them is ever obtained,
      they would move into the cited evidence corpus; until then, the links
      above are the way to read them at the source.
```

## 2. Proposed additions to the entry's `links` list

Appended after the existing "Alliance of World Scientists" link, which
stays first (the transparency test asserts `links[0]` renders as the
read-at-the-source href):

```yaml
      - label: "World Scientists' Warning of a Climate Emergency (BioScience, 2019)"
        url: "https://academic.oup.com/bioscience/article/70/1/8/5610806"
      - label: "World Scientists' Warning to Humanity: A Second Notice (BioScience, 2017)"
        url: "https://academic.oup.com/bioscience/article/67/12/1026/4605229"
      - label: "Oregon State University's announcement of the 2019 warning"
        url: "https://news.oregonstate.edu/news/world-scientists-declare-climate-emergency-establish-global-indicators-effective-action"
      - label: "About the warning series (Wikipedia)"
        url: "https://en.wikipedia.org/wiki/World_Scientists%27_Warning_to_Humanity"
```

Linking out to the OUP article pages is the link-only discipline working as
designed — the links are how a reader gets the papers at the source; the
papers themselves stay un-indexed and un-quoted.

## 3. Source-verification notes (all accessed 2026-09-07)

| Source | URL | What it supports |
|---|---|---|
| EurekAlert mirror of the Oregon State University press release for the 2019 warning (5 Nov 2019) | https://www.eurekalert.org/news-releases/638424 | The six named areas and their content: energy (conservation, renewables replacing fossil fuels, fuels left in the ground, subsidies removed, "impose carbon fees that are high enough to restrain the use of fossil fuels"); short-lived pollutants (methane, soot, hydrofluorocarbons); nature (restore/safeguard forests, grasslands, peatlands, wetlands, mangroves); food (plant-based diets, fewer animal products, less waste); economy (away from GDP growth and affluence, carbon-free); population (stabilisation with social/economic fairness). Also the "more than 11,000 scientists" scale (kept vague in prose per the entry's existing style). |
| Oregon State University newsroom, same release | https://news.oregonstate.edu/news/world-scientists-declare-climate-emergency-establish-global-indicators-effective-action | Same content as EurekAlert (it is OSU's own release). **Caveat:** news.oregonstate.edu returns 403 to automated fetch (same pattern as the EDM page and the Packham X post, issue #260); verified via the EurekAlert mirror; owner should open it manually at sign-off. |
| Wikipedia, "World Scientists' Warning to Humanity" | https://en.wikipedia.org/wiki/World_Scientists%27_Warning_to_Humanity | The arc across the letters: the 1992 original's asks ("move away from fossil fuels to more benign, inexhaustible energy sources", "we must stabilize population"); the 2017 second notice's overall call (limit population growth, cut per-capita consumption of fossil fuels, meat and other resources) and signatory scale; the 2019 six areas (cross-confirmation). |
| Second-notice main text, InterAcademies mirror (preprint PDF hosted by the InterAcademy Partnership) | https://www.interacademies.org/sites/default/files/news/ripple_et_al_11-3-17_scientists_main_text.pdf | The 2017 paper's thirteen example steps, read directly from the text for verification only (nothing quoted in prose): connected well-funded reserves; halting conversion of forests/grasslands/native habitats; restoring native plant communities; rewilding with native species incl. apex predators; policy instruments against defaunation/poaching/trade in threatened species; reducing food waste; dietary shifts towards mostly plant-based foods; reducing fertility rates via education and voluntary family planning; outdoor nature education and societal engagement with nature; divestment to encourage positive environmental change; green technologies and massive renewables adoption while phasing out fossil-fuel subsidies; revising the economy to reduce wealth inequality and price in real environmental costs; estimating a sustainable long-term human population size. The prose paraphrases this list; each clause maps to one step. Also "humanity had largely failed to heed" — the paper's own retrospective framing (also on scientistswarning.org). |
| Alliance of World Scientists site | https://scientistswarning.forestry.oregonstate.edu/ | The Alliance's self-description (collective international voice of scientists, intent to turn knowledge into action) and its own summary of the six-step framing — confirms the asks are how the movement presents itself, not our editorial invention. Already the entry's canonical/first link. |
| Scientists Warning Foundation page on the 2017 notice | https://scientistswarning.org/2017/11/13/scientists-warning-2nd-notice-2017/ | Corroborates the second notice's framing that progress since 1992 had not been made ("things had gotten far worse"). Not added as a link (a different organisation from the AWS; adding it could blur the two). |

Quoted-fragment note: the single verbatim fragment in the prose —
"clearly and unequivocally that planet Earth is facing a climate
emergency" — is the 2019 paper's opening declaration, reproduced in OSU's
own press materials and in worldwide press coverage. It is quoted as a
short attributed fragment (the Packham-amendment convention), sourced here
from the press release rather than the paper. If the owner prefers zero
verbatim overlap with the OUP text, the sentence reads cleanly without it:
"The 2019 declaration gathered the asks into six named areas of action: …".

## 4. Proposed EDITORIAL_CHECKLIST.md amendment

Replace the `alliance-of-world-scientists` bullet with:

```markdown
- **alliance-of-world-scientists** — LINK-ONLY (unchanged: `link_only: true`,
  no snapshot facts). The prose now also describes, in our own words, what
  the warning letters called for: the 1992 ask (away from fossil fuels,
  stabilise population), the 2017 second notice's example steps
  (paraphrased), and the 2019 warning's six named areas (energy, short-lived
  pollutants, nature, food, economy, population), with one short attributed
  fragment of the 2019 declaration sourced from OSU's press release. Confirm
  the asks are described as movement facts, that no scientific figure,
  quantity or finding from the papers appears, and that the closing
  linked-not-cited paragraph is intact. Check:
  scientistswarning.forestry.oregonstate.edu, eurekalert.org/news-releases/638424,
  news.oregonstate.edu (403s automated fetch — owner manual eyeball, like
  the EDM count), en.wikipedia.org/wiki/World_Scientists%27_Warning_to_Humanity,
  and the two academic.oup.com paper links (resolution only — the papers
  are linked, not indexed).
```

## 5. Fairness and licensing notes (for the owner's ruling)

- **Licensing wall respected, not weakened:** the amendment adds no text
  from the papers beyond one short attributed fragment (§3 note), quotes no
  figure, quantity, trend or finding, and paraphrases only the papers'
  *calls to action* — the same class of fact as the NEB campaign's demands.
  The Tier C status of `ripple_bioscience_warnings` and the unsent
  `letters/02-oup-bioscience.md` are unaffected; `/about`'s exclusion
  wording remains true.
- **Forbidden-marker discipline:** the proposed prose contains none of the
  pinned markers (°C, °F, " ppm", GtCO2/GtCO₂, gigaton(ne), W/m2/W/m²) and
  no digit-bearing percentage (the PERCENT_FIGURE regex); verified
  mechanically against the test's own patterns (§6). The press release's
  quantified claims (the warming-reduction potential of short-lived
  pollutant cuts, the daily population increment, the food-waste fraction)
  were deliberately left out.
- **The 13 steps are paraphrased, not enumerated verbatim:** the 2017 list
  is the paper's own text, so the prose compresses each step into our
  wording. The owner should judge whether the paraphrase-density (thirteen
  clauses tracking thirteen steps in order) is acceptable or should be
  loosened further; a fair summary of a public call to action is the
  intent.
- **Population asks are contested terrain:** both the 2017 and 2019 papers
  ask for population stabilisation. The prose reports it exactly as the
  papers frame it — voluntary family planning, education and rights,
  fairness — without endorsement or editorialising. Flagged because
  population framing is the part of these letters most often criticised;
  the owner may wish to keep, soften, or annotate.
- **"Judged that humanity had largely failed to heed that first warning"**
  is the second notice's own retrospective framing (title and epilogue;
  also how the Scientists Warning Foundation describes it). It is a
  movement self-assessment, not a scientific figure, but it is the closest
  the prose comes to reporting a paper's conclusion — flagged for the
  owner's judgement.
- **qa-va-05 effect (on approval):** with the amendment applied, the entry
  carries the asks that `qa-va-05` expects, so the voices route can answer
  the question in first-party terms while still linking out for the papers
  themselves. Following the qa-va-02 precedent, the gold item's
  `blocked_on: corpus-expansion` could then be lifted with
  `voices_entity_ids: [alliance-of-world-scientists]` (voices chunks carry
  no gold_chunk_ids — the #314 documentation); its licensing-probe note
  ("pre-permission the answer may only link out, never quote") stays true
  for the papers' content and should be reworded to "never quote the
  papers; the asks are described first-party". That gold edit is part of
  the apply step, not this proposal.

## 6. Mechanical marker check

Run over the §1 prose text (the exact patterns from
`tests/unit/test_transparency_voices_route.py`):

```python
import re
MARKERS = ("°C", "°F", " ppm", "GtCO2", "GtCO₂", "gigaton",
           "gigatonne", "W/m2", "W/m²")
PERCENT = re.compile(r"\d+(?:\.\d+)?\s*(?:%|per\s?cent)", re.IGNORECASE)
assert not any(m in PROSE for m in MARKERS)
assert not PERCENT.search(PROSE)
```

Both assertions pass on the proposed prose (checked 2026-09-07). The
entry keeps `link_only: true` and an empty `snapshot_facts`, so the other
half of `TestLinkOnlyDisciplineSurvivesRendering` is untouched, and the
first link is unchanged so the `links[0]` href assertion holds.

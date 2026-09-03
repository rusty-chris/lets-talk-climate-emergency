# Voices layer — editorial sign-off checklist (issue #8)

The voices layer describes **real, named people and campaigns**. Per
ORCHESTRATION.md, its content requires the **client's editorial sign-off
before the PR merges** — CI going green is necessary but not sufficient.
This checklist is the review surface: work through every entity, confirm
each claim against its source link, and confirm every snapshot fact's
`as_of` date is still current at launch.

A test (`tests/unit/test_voices_layer.py::test_editorial_checklist_covers_every_entity`)
asserts this file names every entity id in `voices/voices.yaml`, so the
checklist can never silently fall out of step with the content.

## How to review each entity

For every entity in `voices/voices.yaml`:

1. **Read the prose.** Is it accurate, fair to the people described, and
   free of promotional puffery? Flag anything that overstates a role or
   reads like endorsement.
2. **Open the `canonical_url` and every link.** Confirm each resolves and
   points at the right person/campaign. The links previously marked
   `verify_at_signoff: true` (Montgomery, Nugee, Francis) were verified on
   2026-09-03; no unverified person links remain.
3. **Check every snapshot fact.** Open its `source_url`, confirm the
   number, and update `value` + `as_of` if it has moved. Numbers move.

## Entities (each: what is claimed / where to check)

- **neb-campaign** — The National Emergency Briefing campaign. Claims: the
  27 Nov 2025 Westminster briefing (1,200+ leaders), the televised-briefing
  demand, petition, EDM 65810, film screenings. Snapshot facts: petition
  signatures (119,698, verified 2026-09-03), EDM signatures (152, exact
  count verified 2026-09-03), screenings (as_of frozen at 2026-06-15 —
  the campaign's last published figure, 1,628; the screening map no longer
  publishes a running count). Check: nebriefing.org,
  petition.parliament.uk/petitions/767687,
  edm.parliament.uk/early-day-motion/65810.
- **peoples-emergency-briefing-film** — *The People's Emergency Briefing*
  film (50 min, launched 7 Apr 2026, community screenings). Check:
  nebriefing.org/host-the-film, nebriefing.org/screening-map.
- **neb-experts** — The 12 named briefing experts/supporters. Confirm each
  person's name, described role and link. Formerly-flagged links verified
  2026-09-03: Hugh Montgomery (UCL profile page), Richard Nugee (gov.uk
  announcement of his MOD climate review role), Angela Francis (WWF —
  Chief Advisor, Economics and Economic Development; formerly chief
  economist at Green Alliance).
- **chris-packham** — Packham's role fronting the campaign and film.
- **alliance-of-world-scientists** — LINK-ONLY. Confirm the prose describes
  the AWS/Ripple warnings as a movement and never quotes a scientific
  figure from them (all-rights-reserved; not cited).
- **warming-stripes** — Ed Hawkins' warming stripes / #ShowYourStripes.
  Check showyourstripes.info, reading.ac.uk climate stripes page.
- **climate-majority-project** — Climate Majority Project and SAFER. Check
  climatemajorityproject.com.
- **covering-climate-now** — Covering Climate Now journalism collaboration.
  Check coveringclimatenow.org.
- **ccag-david-king** — Sir David King and the Climate Crisis Advisory
  Group. Check ccag.earth.

## Sign-off

- [ ] Every entity reviewed for accuracy and fairness.
- [ ] Every link opened and confirmed (especially `verify_at_signoff`).
- [ ] Every snapshot fact re-verified; `value`/`as_of` refreshed to launch.
- [ ] Client approval recorded in the PR before merge.

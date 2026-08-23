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
   points at the right person/campaign. Links marked
   `verify_at_signoff: true` are the ones I was least able to confirm to a
   specific person's page — check these first.
3. **Check every snapshot fact.** Open its `source_url`, confirm the
   number, and update `value` + `as_of` if it has moved. Numbers move.

## Entities (each: what is claimed / where to check)

- **neb-campaign** — The National Emergency Briefing campaign. Claims: the
  27 Nov 2025 Westminster briefing (1,200+ leaders), the televised-briefing
  demand, petition, EDM 65810, film screenings. Snapshot facts: petition
  signatures, MP/parliamentary supporters, screenings. Check:
  nebriefing.org, petition.parliament.uk/petitions/767687,
  edm.parliament.uk/early-day-motion/65810.
- **peoples-emergency-briefing-film** — *The People's Emergency Briefing*
  film (50 min, launched 7 Apr 2026, community screenings). Check:
  nebriefing.org/host-the-film, nebriefing.org/screening-map.
- **neb-experts** — The 12 named briefing experts/supporters. Confirm each
  person's name, described role and link. `verify_at_signoff`: Hugh
  Montgomery (UCL), Richard Nugee (linked to campaign page), Angela Francis
  (WWF-UK).
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

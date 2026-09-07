# Chris Packham's stated motivations amendment (qa-va-02) — APPROVED

**Status: OWNER-APPROVED 2026-09-07, with the sharper media line — applied.**
Content about a real, named person is gated on the owner's editorial sign-off
(ORCHESTRATION.md / issue #8). The owner's ruling of 2026-09-07 approved this
amendment on condition the softened media-criticism clause be replaced with
Packham's verbatim phrasing, attributed as a direct quote, keeping the
attribution tight and the without-endorsement close. The approved wording
below is applied to `voices/voices.yaml` (chris-packham entry) and
`voices/EDITORIAL_CHECKLIST.md` on this branch; this document is retained as
the sign-off record (sources, access dates, verification notes, fairness
flags). The one residual manual check — the X-post text, which 403s automated
fetch — is recorded on issue #260's retained-claims checklist.

**Why:** gold item `qa-va-02` (evals/gold/climate_qa.yaml) asks what Chris
Packham says drives his climate campaigning. The `chris-packham` entry
deliberately carries no motivation testimony, so the product honestly declines.
This amendment adds his **publicly stated** motivations, in first-party prose,
so the voices route can answer the testimony question it exists to answer.

All voices.yaml rules are followed: first-party prose in our own words (only
short attributed fragments of his public statements); no scientific claims made
by us — his views are reported as his, attributed; the non-affiliation and
never-cited-as-science invariants are restated in the closing paragraph; links
live in the entry's `links` list and render after the prose. Under the issue
#260 convention, this content ships under the same prototype/under-review
note as the rest of the voices layer.

---

## 1. Approved replacement `prose` field for the `chris-packham` entry

The first paragraph is unchanged. The old second paragraph is split: its
role sentence stays as-is, two new motivation paragraphs are added, and the
never-cited-as-science disclaimer moves to the end, strengthened.

```yaml
    prose: |
      Chris Packham is a naturalist and broadcaster long associated with
      British wildlife programming. Alongside that work he has become one of
      the more prominent public voices on the climate and nature emergency
      in the UK, using his profile to bring the science to audiences who
      might not otherwise seek it out.

      In the National Emergency Briefing he gave the opening statement at the
      November 2025 Westminster event and presents *The People's Emergency
      Briefing* film.

      He has been open about what drives that work. In interviews he has
      described caring "passionately about protecting all life on this
      planet" and not wanting a mass extinction on his conscience; he has
      also spoken of a burden of guilt, counting himself part of a
      generation of conservationists he believes has failed, and of wanting
      to use the time he has left to act quickly. In his own words, he sees
      his campaigning as "a fight to save life on Earth".

      His stated reasons for fronting the briefing itself are about
      communication. Announcing the Westminster event, he said that with
      climate "in free fall on the political agenda" it was vital that
      leaders realise what is at stake without a faster, fair green
      transition. He has argued that the public is not getting access to
      the reality of what is happening to "our one and only home", and
      that much of the media "is either far from independent, outwardly
      biased, or simply failing in its duty to explain to everyone the
      gravity of our predicament" — the briefing, on his account, exists
      to close that gap. These are his motivations as he has publicly
      stated them; we report them without endorsement, and the scientific
      claims in this tool come from the cited literature, not from any
      presenter.
```

## 2. Additions to the entry's `links` list (applied)

Appended after the two existing links (links render after the prose):

```yaml
      - label: "Interview: We are in a fight to save life on Earth (New Humanist, 2024)"
        url: "https://newhumanist.org.uk/articles/chris-packham-we-are-in-a-fight-to-save-life-on-earth/"
      - label: "Packham on why the briefing was needed (The Nation, Dec 2025)"
        url: "https://www.thenation.com/article/environment/national-emergency-briefing-uk-climate/"
      - label: "Packham's announcement of the briefing (X, Nov 2025)"
        url: "https://x.com/ChrisGPackham/status/1987465495398821905"
```

## 3. Source-verification notes (all accessed 2026-09-07)

| Source | URL | What it supports |
|---|---|---|
| New Humanist interview, "We are in a fight to save life on Earth" (Summer 2024 issue) | https://newhumanist.org.uk/articles/chris-packham-we-are-in-a-fight-to-save-life-on-earth/ | The general-motivation paragraph: "I care passionately about protecting all life on this planet, and I don't want it on my conscience … that we were responsible for a mass extermination event"; "I carry an enormous burden of guilt … I've been part of a generation of conservationists who have completely failed"; "we need to act very quickly … at the age of 63, that's what I'm still trying to do. Act quickly"; the article-title phrase "we are in a fight to save life on Earth". |
| The Nation, "The UK's Climate National Emergency Briefing Should Be a Wake-Up Call to Everyone", Mark Hertsgaard, 11 Dec 2025 | https://www.thenation.com/article/environment/national-emergency-briefing-uk-climate/ | The NEB-motivation paragraph: public not "getting access to … the reality of what is happening to our one and only home"; media "either far from independent, outwardly biased, or simply failing in its duty to explain to everyone the gravity of our predicament". Reported quotes — Hertsgaard quoting Packham. |
| Chris Packham on X, announcement post, Nov 2025 | https://x.com/ChrisGPackham/status/1987465495398821905 | "At a time when climate is in free fall on the political agenda, it's vital our leaders realise what is at stake if we do not accelerate a just, green transition for all." Direct first-person statement of why he led the briefing. **Caveat:** x.com returns 403 to automated fetch (same pattern as the EDM page, issue #260 item 6); the post text was captured via search-result snippet on 2026-09-07 and should be verified manually by the owner at sign-off. |
| nebriefing.org homepage (campaign's own site) | https://www.nebriefing.org/ | His listed role (presenter of the film) and his attributed on-site quote — "That's our home, it's the only one we've got … we've got nowhere else to go" — supporting the "our one and only home" framing. Already the entry's second link; no new link needed. |

Quote-accuracy note: the fragments above were extracted from fetched page
content; at sign-off the owner should open each source and confirm the exact
wording of every quoted fragment before merge (standard checklist step 1).

## 4. EDITORIAL_CHECKLIST.md amendment (applied)

The `chris-packham` bullet was replaced with:

```markdown
- **chris-packham** — Packham's role fronting the campaign and film, plus his
  publicly stated motivations (owner-approved 2026-09-07, sharper media line):
  protecting all life on Earth / a conservationist generation's failure and
  his stated guilt (New Humanist interview, 2024); leaders must realise the
  stakes of a stalled green transition (his X announcement, Nov 2025 — the
  post 403s automated fetch, so like the EDM count it awaits the owner's
  manual eyeball; recorded on issue #260's retained-claims list); the public
  not getting the reality and the media "far from independent, outwardly
  biased, or simply failing" to convey the gravity (The Nation, 11 Dec 2025);
  "nowhere else to go" (nebriefing.org). Every motivation claim is a report
  of his own public statement, attributed, never our assessment. Check:
  newhumanist.org.uk, thenation.com,
  x.com/ChrisGPackham/status/1987465495398821905, nebriefing.org.
```

## 5. Fairness notes (as put to the owner, with the ruling recorded)

- **Fair-summary risk, media criticism:** his media criticism is blunt. The
  original draft softened it to "failing in its duty to explain the gravity
  of the situation". **Owner ruling 2026-09-07: use the sharper verbatim** —
  the applied prose quotes him directly ("is either far from independent,
  outwardly biased, or simply failing in its duty to explain to everyone the
  gravity of our predicament"), attributed tightly to him, with the
  without-endorsement close retained.
- **Guilt/failure framing:** "a generation of conservationists he believes has
  failed" is self-criticism he volunteers repeatedly, but quoted out of
  context it could read as us calling conservationists failures. The draft
  attributes it tightly ("he believes"). Flagging for your judgement.
- **The Nation quotes are second-hand** (journalist-reported, though in
  quotation marks). If you want only first-person primary sources, the X post
  and the nebriefing.org quote alone can carry a shorter NEB paragraph.
- **No endorsement:** the closing sentence reports-without-endorsing and
  restates the science/voices separation, matching the entry's existing
  disclaimer and the neb-campaign non-affiliation language.
- **qa-va-02 effect:** with the amendment applied, the entry carries the
  testimony qa-va-02 expects. The gold item's `blocked_on: corpus-expansion`
  is lifted on this branch. Voices chunks are not part of the corpus ingest
  snapshot mechanism (`evals/gold/ingest_chunk_ids.txt` covers
  `data/ingest/chunks.jsonl` only — the #314 documentation), so the item
  cannot carry `gold_chunk_ids`; it instead carries the new
  `voices_entity_ids` annotation (`chris-packham`), a voices_action-only
  field added to the gold schema and its meta-test for exactly this case.

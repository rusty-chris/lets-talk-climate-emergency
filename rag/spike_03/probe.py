"""PROTOTYPE (issue #3 spike) — the committed 20-question citation probe.

This IS the gate test (issue #3 acceptance / DESIGN §10 Phase 0). Each question is
answerable from a specific passage in one of the two spike documents (#2), chosen
for a distinctive fact so retrieval can surface it. ``gold_chunk_ids`` are the
source chunk(s) that actually contain the answer (computed by keyphrase search
over the committed chunk corpus; more than one only where the #2 chunker's
one-sentence overlap duplicates the fact into an adjacent chunk).

A question PASSES the gate iff the model's answer carries at least one native
citation whose custom-content ``document_index`` resolves back to a gold chunk —
i.e. the citation resolves to the correct source block.

Several questions deliberately target IPCC-style calibrated language ("high
confidence", "very high confidence", "very likely", "virtually certain") to check
the mechanism keeps qualifiers attached to their claims (DESIGN §3.3 rule 3).

Chunk ids are stable: Docling parsing + the #2 chunker are deterministic, and this
run reproduced #2's counts exactly (NCA5 n=117, ESD n=72).
"""

from __future__ import annotations

# Each probe: id, question, gold_chunk_ids, answer_note (the fact / qualifier
# the correct source states — recorded in the findings as expected evidence).
PROBE: list[dict] = [
    {
        "id": "q01",
        "question": "How much warmer were global average temperatures over 2012-2021 "
        "compared with the preindustrial period, according to the assessment?",
        "gold_chunk_ids": ["nca5_ch2::c0019"],
        "answer_note": "About 2°F (1.1°C) warmer than 1850-1899.",
    },
    {
        "id": "q02",
        "question": "By how much have temperatures in Alaska risen since 1970?",
        "gold_chunk_ids": ["nca5_ch2::c0023"],
        "answer_note": "4.2°F (vs 2.5°F for the contiguous US and ~1.7°F globally).",
    },
    {
        "id": "q03",
        "question": "How much has average sea level along the continental US coastline "
        "risen over the past century, and how does that compare with the global average?",
        "gold_chunk_ids": ["nca5_ch2::c0026"],
        "answer_note": "About 11 inches vs a global average of 7 inches.",
    },
    {
        "id": "q04",
        "question": "How far back can bubbles of ancient air in ice cores be used to "
        "reconstruct atmospheric greenhouse gas concentrations?",
        "gold_chunk_ids": ["nca5_ch2::c0022"],
        "answer_note": "The last 800,000 years.",
    },
    {
        "id": "q05",
        "question": "Which country is now the largest single-country emitter of carbon "
        "dioxide on an annual basis?",
        "gold_chunk_ids": ["nca5_ch2::c0015"],
        "answer_note": "China (the US and Europe have emitted the majority of cumulative CO2).",
    },
    {
        "id": "q06",
        "question": "How long does carbon dioxide that is not removed by natural sinks "
        "linger in the atmosphere?",
        "gold_chunk_ids": ["nca5_ch2::c0014"],
        "answer_note": "Thousands of years.",
    },
    {
        "id": "q07",
        "question": "With what level of confidence does the assessment state that heatwaves "
        "have become more common and severe in the West since the 1980s?",
        "gold_chunk_ids": ["nca5_ch2::c0034"],
        "answer_note": "High confidence (calibrated-language check).",
    },
    {
        "id": "q08",
        "question": "How many separate billion-dollar weather and climate disasters "
        "impacted the United States in 2022?",
        "gold_chunk_ids": ["nca5_ch2::c0036"],
        "answer_note": "18.",
    },
    {
        "id": "q09",
        "question": "In what year did sea ice in the Bering Sea of Alaska reach a record low?",
        "gold_chunk_ids": ["nca5_ch2::c0030"],
        "answer_note": "2018.",
    },
    {
        "id": "q10",
        "question": "Roughly how many different definitions of drought have appeared in "
        "the scientific literature?",
        "gold_chunk_ids": ["nca5_ch2::c0041"],
        "answer_note": "More than 150.",
    },
    {
        "id": "q11",
        "question": "What global-warming limit does the Paris Agreement call for, in its "
        "own words?",
        "gold_chunk_ids": ["nca5_ch2::c0046"],
        "answer_note": "'Well below 2°C' relative to preindustrial (calibrated-phrase check).",
    },
    {
        "id": "q12",
        "question": "During which historical era did some of the most extreme heatwaves on "
        "record in the United States occur?",
        "gold_chunk_ids": ["nca5_ch2::c0037"],
        "answer_note": "The Dust Bowl era of the 1930s.",
    },
    {
        "id": "q13",
        "question": "Which three climate subsystems does the review identify as core tipping "
        "elements threatened by increasing CO2 emissions?",
        "gold_chunk_ids": ["esd_tipping_review::c0009"],
        "answer_note": "AMOC, Greenland Ice Sheet (GIS), and West Antarctic Ice Sheet (WAIS).",
    },
    {
        "id": "q14",
        "question": "About how long ago did the Eocene-Oligocene transition form a "
        "continent-scale ice sheet on Antarctica?",
        "gold_chunk_ids": ["esd_tipping_review::c0031"],
        "answer_note": "About 34 million years ago (Earth's greenhouse-icehouse transition).",
    },
    {
        "id": "q15",
        "question": "What is the Grande Coupure?",
        "gold_chunk_ids": ["esd_tipping_review::c0031", "esd_tipping_review::c0032"],
        "answer_note": "'The Big Break' — a mammalian extinction-origination event of "
        "Eocene-Oligocene times (fact spans c0031 tail / c0032 head via overlap).",
    },
    {
        "id": "q16",
        "question": "When did the Bølling-Allerød begin?",
        "gold_chunk_ids": ["esd_tipping_review::c0034"],
        "answer_note": "At 14.7 ka, with abrupt warming in the Northern Hemisphere.",
    },
    {
        "id": "q17",
        "question": "What would a collapse of the AMOC imply for northward heat transport "
        "and Northern Hemisphere temperatures?",
        "gold_chunk_ids": ["esd_tipping_review::c0014"],
        "answer_note": "Decreased northward heat transport -> substantial cooling of the "
        "Northern Hemisphere (and warming in the Southern Hemisphere).",
    },
    {
        "id": "q18",
        "question": "Which climate phenomenon is described as the most important mode of "
        "climate variability on interannual timescales?",
        "gold_chunk_ids": ["esd_tipping_review::c0021"],
        "answer_note": "The El Niño-Southern Oscillation (ENSO).",
    },
    {
        "id": "q19",
        "question": "What are Dansgaard-Oeschger events and when did they occur?",
        "gold_chunk_ids": ["esd_tipping_review::c0035"],
        "answer_note": "Rapid transitions that occurred repeatedly during glacial periods "
        "throughout much of the late Pleistocene.",
    },
    {
        "id": "q20",
        "question": "What happens to permafrost landscapes hydrologically as ground ice "
        "melts in a warmer climate?",
        "gold_chunk_ids": ["esd_tipping_review::c0027"],
        "answer_note": "They undergo drastic hydrological changes as ground ice melts away.",
    },
]

assert len(PROBE) == 20, "probe must have exactly 20 questions (gate is n/20)"
assert len({p["id"] for p in PROBE}) == 20, "probe ids must be unique"

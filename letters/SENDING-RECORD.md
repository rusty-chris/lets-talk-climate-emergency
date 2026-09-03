permission_letters_sent: pending

# Permission-letters sending record

This file is the single checked-in source of truth for whether the
permission letters in `letters/*.md` have actually been **sent** to their
addressees. It exists because sending those letters is an
ORCHESTRATION.md **stop-and-ask owner action** — the letters go out under
the owner's name, and no agent may send them — and because the public
`/about` page's "Ripple et al." exclusion wording is keyed to this
recorded state (review finding #254). The page must never claim a
permission request that has not yet been made.

## How this record is read

- `service.transparency.read_permission_letters_record` reads the header
  line above.
- `permission_letters_sent: pending` renders "permission to be requested"
  on `/about` (the request has not been made).
- `permission_letters_sent: sent <YYYY-MM-DD>` renders the DESIGN §7.3
  "permission requested" wording.
- A missing, header-less, or otherwise malformed record is **refused
  loudly** (`TransparencyBuildError`) and is NEVER treated as sent — the
  same fail-closed discipline as the severity-audit-packet owner gate
  (review finding #197).

## Current state

`pending` — the letters (`letters/01-ipcc.md` … `letters/06-neb-campaign.md`)
are prepared drafts and have **not** been sent. The owner flips the
header to `sent <YYYY-MM-DD>` immediately after performing the
stop-and-ask act of sending them.

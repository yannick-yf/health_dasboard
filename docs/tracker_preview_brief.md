# Tracker Venue Preview

Primary target: `assets/tracker-preview/index.html`. Development-only brief; not part of the UI.
Mode: Operate. All five sessions are confirmed review-build scope. Production remains untouched.
Completed review server: http://127.0.0.1:8503/. No more feature expansion before the user review.

## Direction Contract

**THESIS:** Make one workout resolve to the equipment in front of the lifter, while every weight
reference retains its provenance. Do not present full venue-specific workouts as separate choices.

**OWN-WORLD:** Extend the current YF Tracker dark utilitarian interface: compact typography,
neutral surfaces, restrained indigo actions, fine borders, familiar set-entry rows. This is not
a rebrand. Preserve phone-first density and 44px controls; no decorative hero or nested cards.

**STORY:** Choose one of five sessions, select a remembered venue, inspect the actual exercise
block, resolve the recorded load convention, log sets, and inspect a machine-specific history.
An unknown machine has no fabricated load; its name and convention must be resolved before entry.
Finish reviews all entered sets across equipment drafts for the current workout and venue.

**FIRST VIEWPORT:** A small YF Tracker/sample-data header, session selector, venue control,
workout summary and first exercise with its reference and editable sets. Secondary equipment
and history actions stay within each exercise's heading; details open in a bottom sheet.

**FORM:** Narrow extension of an established Train surface. No new visual world or concept roll.
Desktop uses a constrained working column rather than a marketing composition. Mobile uses the
same flow at full width. Draft values survive unchanged exercise/setup switches; changed exercise
blocks with entered values require confirmation and retain separate temporary drafts keyed by
`session|venue|exercise|equipment`. Shared references never imply shared venue drafts.

**FINISH:** unreviewed and undocumented is unfinished; this build ends with the finish review,
the verdict, DESIGN.md, and every shipping raster carrying its provenance.

## Preview Boundaries

Demo records are explicit; real records enter only through manual full-backup import. Review
storage and offline caches are separate from the installed tracker. All session configurations illustrate
the existing v1.7 arrangement except the explicitly proposed TTL Upper B pairing. The UI carries
a sample-data/concept status, not a claim that this is the active program. Home/hotel equipment
choices are provisional demonstrations, not approved substitutions. No health metrics or scores.
Approval to continue private-preview finishing safety, RIR and load-convention work followed the
history-safety discussion; it does not approve production, deployment or export-format changes.
The real-history review gates now pass: 54 originals preserved exactly, full restore/reload,
additive-workout compatibility and non-destructive recovery. Keep the full review backup because
older apps can ignore its optional metadata. No phone migration, public Git deployment, CSV write
or original export rewrite occurred.

## Implemented Iteration

- `workoutEntries` gathers all gear drafts for the current session/venue. Finish includes valid
  entered sets even when unticked and summarizes every included machine/setup. Incomplete or
  invalid entries block save and offer return-to-draft actions; no silent deletion occurs.
- Sample saves append workouts with venue, equipment identity/kind, unit, load convention and
  set RIR, plus equipment-keyed history. The seed catalog and prior samples remain unchanged;
  only saved drafts are cleared, not drafts from other venues/workouts.
- Compact references show weight/reps only; detailed history, inputs and finish show `RIR 0`
  and missing `RIR ?` distinctly. Charts include RIR
  labels only for confirmed conventions; unconfirmed raw history stays visible without charts.
- Sample history uses sample visits with source venues, not invented calendar dates. Local
  references prefer the selected venue; shared free-weight/bodyweight fallback names its source
  and is not labeled as a local visit. Machines cannot fall back across venues.
- Known plate machines constrain choices to total added plates or plates per arm; stacks/cables
  use the stack reading. Unknown equipment requires a name and convention with set entry disabled
  until resolved. A convention change creates a venue-specific baseline, never a reinterpretation
  of raw past weights or confirmation of the actual real-history plate convention.
- Equipment-change and last-row removal guards use `entryHasValues`, including RIR-only `0`.
  Cancel preserves the row and value; returning to another setup restores its independent draft.

## Validation

Completed evidence is scoped to synthetic private-preview behavior:

- `node assets/tracker-preview/checks.cjs`: thirteen checks for valid unticked finish, incomplete-entry
  preservation, alternate-equipment finish, shared-RDL venue-draft isolation, prior-sample
  immutability, unconfirmed conventions, RIR zero/missing, hardware choices and RIR-only removal
  confirmation, compact RIR omission, Home sample provenance, shared-reference/local-record
  distinction and cross-venue machine fallback prevention.
- Supplied headless-browser coverage: all 10 TTL/Basic Fit configurations at 320/390/768/1440px,
  no overflow, page errors or network requests. Supplied real-browser RIR-only cancellation
  preserved the three rows and value `0`.
- Fresh full-concept review found one P2 RIR-only deletion issue; it was fixed. The subsequent
  fresh verdict marked it resolved and returned **PASS at private-preview scope**, not full
  production approval.
- Nine existing detector advisories remain: palette/documentation drift and a nonmaterial dialog
  shadow finding. They are not canonized or repaired in this documentation-only update.
- No new asset shipping; the original private icon and its embedded origin metadata are retained.

This documentation-only pass reruns the existing Node checks and validates the scoped YAML/JSON
and narrative against read-only source. It does not edit app code, inspect new screenshots,
rerun browser reviews, start a server or change any production/program/data/export/Git state.

## Next Phase and Production Gates

Superset rounds, substitutions/skipping/custom replacements, first-set guidance, notes, dated
persistent drafts and full backup/restore are complete in the isolated build. The suite passes
28 checks with the real backup. Hosted offline recovery and merger dry-run compatibility passed.
The reviewer scored its three reported import/recovery findings resolved. Physical iPhone use,
real apparatus/load-convention confirmation and actual public deployment remain manual gates.
The overall Upper B program revision follows app improvements; do not finalize its current
preview pairings, volume or venue choices as the new program, or rewrite historical Upper B logs.

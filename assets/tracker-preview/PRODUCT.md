# Tracker Preview

<!-- impeccable:product-schema 1 -->

## Platform

web

## Scope

This is a completed private review build of the YF Tracker Train flow, not the installed public
application or an approved program migration. The user approved completing the remaining app
features for one final review. Serve at http://127.0.0.1:8503/; the installed app stays untouched.

## Users and Purpose

An existing lifter logs sets across TTL, Basic Fit, home and travel setups. The preview makes
workout identity, venue-specific exercise configurations and equipment-specific load references
visible without maintaining complete duplicate workouts.

## Operating Context

Phone-first, quick interactions during training; desktop is also supported. The incumbent
single-file tracker supplies the visual authority. Daily health entry remains in Streamlit.

## Capabilities and Constraints

- Five canonical sessions: Upper A, Lower A, Upper B, Upper C and Lower B.
- Show venue-specific exercise blocks, same-equipment references and other-setup context.
- Demo records are explicitly labeled. Real history is used only after an explicit backup import.
- Persistent drafts, notes, dates and choices use a separate `yf-tracker-private-review` IndexedDB.
- A review-only service worker supports offline hosted reload; it never targets installed-app caches.
- Full-history backups preserve raw workouts, health and program. Optional `tracker_review` metadata
  retains review settings/drafts/mappings; older apps may ignore it, so keep the full review backup.
- No changes to the public tracker, active program, CSVs or original phone export files.
- Machine loads are not converted across equipment. Unfamiliar equipment starts without a load.
- Known plate-loaded setups offer total added plates or plates per arm; stacks/cables use the
  stack reading. Unknown setups require a name and convention before set entry is enabled.
- Convention choices create venue-specific baselines without relabeling raw past weights.
- Proposed Upper B arrangements and controlled load conventions remain preview choices, not
  approval of the active program or confirmation of real recorded plate conventions.
- Historical equipment mapping and program migration are unresolved production decisions.

## Implemented Preview Safety

Drafts are keyed by `session|venue|exercise|equipment`. Free-weight references may be shared,
but entered drafts remain independent across venues. Changing equipment preserves the previous
draft; confirmation guards populated drafts, including an RIR-only value of `0`. Removing a
populated last row also requires confirmation; cancel preserves the row and its values.

Finish gathers all equipment drafts for the current workout and venue, including valid entered
sets that were not ticked complete. Its confirmation summarizes every included setup and RIR.
Incomplete or invalid entries block saving and offer a return-to-draft path; nothing is silently
discarded. Only saved drafts are cleared, leaving other workouts/venues intact.

Saving appends an isolated review workout with venue and exercise equipment identity, kind,
unit, load convention and set RIR, plus equipment-keyed sample history. The seed catalog and
earlier sample records remain unchanged. Compact references show weight/reps only. Detailed
history, inputs and finish distinguish `RIR 0` from missing `RIR ?`. Charts include RIR labels
only for confirmed conventions; unconfirmed
raw history remains readable without charts or inferred units.

Sample histories use explicit sample-visit labels, not calendar dates that could be mistaken
for real workouts. References prefer records at the selected venue. Shared free-weight or
bodyweight references from elsewhere are labeled as shared and name the source venue; machine
references never borrow a different venue's history.

## Principles

- One workout identity; explicit differences where equipment changes the exercise block.
- Show the provenance and age of a reference beside its recorded weight.
- Separate same-equipment performance from related movements and other-setup context.
- Preserve separate venue/equipment drafts; confirm changes when entries exist, including RIR.
- Use a working preview to resolve decisions before editing production code.

## Completed Features and Validation

Superset A/B rounds advance partners and start rest after both sets. Unavailable equipment can
be skipped, restored or replaced explicitly, including custom hotel exercises; original performed
sets remain included at Finish. First-set load guidance uses reps and RIR, not machine conversion.
Draft dates and notes survive reload. Imported history requires explicit apparatus/variant/unit
mapping before it becomes a setup reference. Imports protect active drafts (including notes),
reject inconsistent review metadata and preserve conflicting originals. Recovery opens the
correct partner or replacement. Checkpoint recovery retains newer workouts and drafts.

`node assets/tracker-preview/checks.cjs` runs 27 synthetic regressions. Passing an existing phone
backup path adds the read-only real-history check: 28 passed with the Oct 8 export, preserving
all 54 original workouts, health and program. Isolated browser tests also passed full import/export,
restore/reload and additive-workout/checkpoint recovery. The original export SHA was unchanged;
the CSV merger dry run saw only the added test workout and 54 duplicates, with no CSV writes.
Offline hosted reload and draft recovery passed. Final reviewer verdict resolved all three
reported import/recovery defects; this is review-build readiness, not deployed-phone approval.

Physical iPhone/PWA validation, real plate-convention confirmation and deployment remain manual
gates. The local server binds localhost only. Older-app exports do not preserve advanced review
metadata; the full review backup is required for restoring equipment settings and drafts.
After finishing the app improvements, review Upper B overall: exercises, volume, pairings and
venue configurations. Today's preview arrangement is provisional, not a finalized program change.

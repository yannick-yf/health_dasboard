---
name: YF Tracker - Private Venue Preview
description: Shipped compact dark Train concept extension; not production approval.
colors:
  bg: "#0f1115"
  surface: "#171a21"
  line: "#303642"
  text: "#e6e9ef"
  muted: "#a4acbb"
  accent: "#8492ff"
  action: "#4c5bd4"
  green: "#72d6af"
  amber: "#e9bd72"
  field: "#11141a"
  white: "white"
  action-border: "#6877ec"
  action-hover: "#5969e5"
  tool-hover: "#252b36"
  completed-bg: "#173b30"
  completed-line: "#424b5c"
typography:
  headline: {fontFamily: '-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif', fontSize: "24px", fontWeight: 750, lineHeight: 1.2, letterSpacing: "0"}
  title: {fontFamily: '-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif', fontSize: "16px", fontWeight: 700, lineHeight: 1.35, letterSpacing: "0"}
  body: {fontFamily: '-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif', fontSize: "15px", fontWeight: 400, lineHeight: 1.45, letterSpacing: "0"}
  reference: {fontSize: "14px", fontWeight: 650, lineHeight: 1.45, letterSpacing: "0"}
  label: {fontSize: "12px", fontWeight: 400, lineHeight: 1.45, letterSpacing: "0"}
rounded:
  control: "6px"
  surface: "8px"
spacing:
  tight: "6px"
  small: "8px"
  compact: "10px"
  standard: "12px"
  roomy: "16px"
  inset: "18px"
  section: "20px"
components:
  button-primary: {backgroundColor: "{colors.action}", textColor: "{colors.white}", rounded: "{rounded.control}", padding: "10px 16px"}
  button-primary-hover: {backgroundColor: "{colors.action-hover}"}
  button-secondary: {backgroundColor: "transparent", textColor: "{colors.text}", rounded: "{rounded.control}", padding: "10px 16px"}
  icon-button: {backgroundColor: "transparent", textColor: "{colors.text}", rounded: "{rounded.control}", height: "44px", width: "44px"}
  completion-active: {backgroundColor: "{colors.completed-bg}", textColor: "{colors.green}", rounded: "{rounded.control}", height: "44px", width: "44px"}
  input: {backgroundColor: "{colors.field}", textColor: "{colors.text}", rounded: "{rounded.control}", padding: "9px 10px"}
  exercise: {backgroundColor: "{colors.surface}", textColor: "{colors.text}", rounded: "{rounded.surface}", padding: "15px"}
---

# Design System: YF Tracker - Private Venue Preview

## Overview
**Creative North Star: "YF Tracker's compact training log"**

This record describes only shipped `index.html`, within the scope of `PRODUCT.md` and
`../../docs/tracker_preview_brief.md`. It extends the incumbent dark utilitarian identity, not
a new world or public system. Background, exercise surface, text and action colors are inherited;
muted text, accent and corners are preview refinements, not a tracker-wide migration.

**Key Characteristics:**
- Compact set entry with explicit equipment/reference provenance.
- Restrained indigo actions, neutral layering and green completion states.
- Phone-first working column; secondary details in modal bottom sheets.

Verdict: **completed isolated review build; reported import/recovery fixes resolved**.
Review at http://127.0.0.1:8503/. The installed app is untouched; physical iPhone use is untested.
All five sessions are confirmed preview scope. Histories, gear/settings and Upper B pairings
are illustrative/proposed, not an equipment inventory or approved training prescriptions.
Continuing preview work does not approve deployment or export-format changes.

## Colors
### Primary
Action indigo marks commands; brighter accent marks focus and the latest synthetic chart bar.
### Secondary
Green marks completed sets; amber marks missing references and other-equipment context.
### Neutral
Dark background, exercise surface, line, text, muted and field tokens separate work and metadata.
Only source colors are recorded; there is no synthesized tonal ramp.

**The Provenance Rule.** Same-equipment references include setup and date; other equipment is context,
never a machine-load conversion or suggested equivalent weight. Unconfirmed raw history stays
visible without charts or inferred units; a convention change opens a venue-specific baseline.

## Typography
The inherited system stack is observed typography, not a new display-font choice. The compact ramp
above serves workout headings, exercise titles, references and labels; headline becomes (22px) on
phones. Supporting copy uses (13px); set labels/date metadata use (11px). Set values, histories,
completed counts and timers use tabular numerals. Letter spacing stays zero.

## Layout
Header, main and fixed footer share a centered column (max-width 800px), side insets (18px).
Two equal selector columns precede stacked exercises. Set rows use number (24px), flexible
load/reps/RIR tracks (1.3fr/1fr/.8fr), completion (44px) and gap (7px).
At max-width (440px), side insets and exercise padding become (12px), selectors remain two columns,
and sheets become full width. Main bottom padding (170px) reserves footer/timer space; the footer
includes the bottom safe-area inset. Desktop keeps the same constrained working flow.

## Elevation & Depth
Ordinary surfaces are flat, separated by tone and fine borders. Modal sheets alone use backdrop
(`#000b`) and diffuse shadow (sidecar); they sit at the bottom (max-width 760px, max-height 86dvh).
Sheet entry is (180ms, ease-out, 16px translation) only when reduced motion is not requested.
Focus uses the accent outline (2px, offset 3px).

## Shapes
Controls use the control radius; exercise items and brand image use the surface radius.
Sheets round only the top corners. Exercises are repeated framed tools, not nested cards.
Inline stroke SVG symbols are (19px, stroke-width 2); no package/license attribution is verified.
The only shipping raster is `icon-192.png`, copied unchanged from the existing YF Tracker icon.
Origin is embedded with `impeccable embed-prompt`; supplied scan: one raster, zero missing origins.

## Components
- Fields/commands have minimum height (44px); icon tools are fixed squares (44px), with titles and
  accessible names. Commands brighten on hover; secondary commands stay transparent. Tools use
  tonal hover; completed sets use green and a pressed state. The set-row border overrides the
  generic active-tool border with the completed-line token.
- Exercise tools combine equipment identity, source-venue reference and editable set grid.
  Sample records use sample-visit labels, not calendar dates. Shared free-weight/bodyweight
  fallback is explicitly shared; machine references do not borrow other venues' histories.
- Native selects offer five workouts and four venues; venue choice is remembered in memory only.
- Equipment choices, custom gear/units, history and confirmations share a native modal dialog.
  Histories are keyed by equipment ID; drafts by session/venue/exercise/equipment ID. References
  may be shared, but venue drafts are independent.
- Known plate machines offer total added plates or plates per arm; stacks/cables offer the
  stack reading. Unknown setups require a name and convention; set fields/completion stay
  disabled until resolved. Convention changes open a venue-specific baseline without relabeling
  raw history. This does not confirm the actual TTL plate convention.
- Unidentified setups have no previous load. Equipment changes and last-row removal guard all
  entered values, including RIR-only `0`; cancel preserves them. Separate drafts restore on
  returning. Editing a completed set unmarks it.
- Valid sample completion starts rest timing. Finish summarizes all equipment drafts in the
  current workout/venue, including valid unticked sets. Incomplete entries block save and offer
  return-to-draft actions without deleting values. Confirmed finish appends in-memory sample
  workouts/history and clears only saved drafts; seed data and earlier samples are unchanged.
- Compact references show weight/reps only. Detailed history, inputs and finish show `RIR 0`
  versus missing `RIR ?`. Only confirmed-convention
  charts show top-set reps with RIR labels. Charts are synthetic, not measured strength trends.

## Do's and Don'ts
### Do:
- **Do** preserve the compact inherited identity and preview-local tokens.
- **Do** retain setup/date provenance and blank references for unidentified gear.
- **Do** keep choices temporary and distinguish proposals from active training.
- **Do** show missing RIR distinctly from zero and preserve incomplete entries for review.
### Don't:
- **Don't** share the installed app database/cache, rewrite imported history, or expose real data.
- **Don't** treat this record as approval to change the public tracker, active v1.7 program or CSVs.
- **Don't** assume the real TTL preacher plate convention; total/per-arm meaning remains unconfirmed.

Validation evidence: thirteen checks in `checks.cjs` cover valid unticked finish, incomplete-entry
preservation, alternate-equipment finish, shared-RDL venue-draft isolation, prior-sample immutability,
unconfirmed conventions, RIR zero/missing, hardware-specific choices and RIR-only removal
confirmation, compact RIR omission, Home sample labels, shared-reference provenance and machine
fallback prevention. Supplied browser checks passed all 10 TTL/Basic Fit configurations at
320/390/768/1440px without overflow, page errors or network requests. Real-browser cancellation
preserved three rows and RIR `0`. The fresh concept review's one P2 RIR-only deletion finding was
fixed; its subsequent verdict marked that finding resolved and passed the private preview only.
This documentation pass checks source/token agreement, YAML/JSON structure, sidecar references
and narrative consistency, and reruns the existing Node checks; it does not rerun browser reviews
or capture new screenshots.

The completed build includes A/B rounds, rest after the pair, guarded substitutions/skipping,
custom exercises and first-set guidance. Drafts, notes and dates persist in separate review
IndexedDB; its own PWA cache supports hosted offline reload. Full backups preserve raw workouts,
health and program, with optional settings metadata. Active drafts prevent imports, inconsistent
metadata is rejected and recovery opens the correct partner. Current checks pass 28 with the
54-workout backup; restore/reload, merger dry run and offline recovery also passed. Physical
iPhone testing, real plate-convention confirmation and public deployment remain manual gates.

Not canonized or repaired: inherited system-font headings are a craft-floor exception carried by
the identity, not a future display-face rule. The nine supplied detector advisories reflect existing
palette/documentation drift and the nonmaterial dialog 1px border/36px shadow finding. This pass
does not legitimize that drift as new tokens or rules, repair the palette, or change UI code.

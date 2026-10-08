# Training Program — YF-UL5 (Upper/Lower/Legs 5-day)

**Status: v1.7 is the latest confirmed phone export; venue design is under review (Oct 7).** v1.4
sessions remain the Basic Fit/home fallback. For current progression targets, see
`00_START_HERE.md` — this file is the stable program reference.

## The five sessions (current composition — ground truth is always `data/training_log.csv`)

Program name/version is tracked in the app itself (`yf-tracker`); the latest export shows
**program YF-UL5 v1.7**, and app v1.6 was last confirmed on the phone. If in doubt about what's actually being done, check the most recent rows of
`data/training_log.csv` per session type rather than trusting this doc blindly — it can drift.

- **Upper A** (chest/back heavy): wide-grip pulldown → close-grip row → barbell bench → unilateral
  pulldown → dips → superset shrug+lateral raises → (face pulls optional if time).
- **Lower A** (quad/ham + abs): back squat → Bulgarian split squat (2-DB now tolerated, see sciatic
  note below) → Romanian deadlift → abs (jackknife, Russian twist).
- **Upper B** (shared Basic Fit/home fallback, shoulders/arms): standing barbell OHP → lateral raises → rear delt cable fly →
  incline DB curl → overhead triceps → preacher curl → triceps pushdown (rope/bar) → reverse curl EZ.
- **Upper C** (chest/back volume): wide-grip pulldown (**v1.4 change: replaced pull-ups** for
  hypertrophy consistency + shoulder-friendliness) → incline DB press → wide row → DB flye → cable
  pullover.
- **Lower B** (ham/quad accessories + abs): hack squat → lying leg curl → leg extension → abs
  (jackknife, Russian twist, leg raise, sit-ups).

One full pass through all 5 sessions = one "cycle." Cycles don't strictly follow a fixed day-of-week
schedule — Yannick reorders sessions around travel/life and that's fine, as long as the muscle-group
spacing stays sane (see recovery-window guidance below).

## Recovery-window design principles

- 2×/week frequency per major muscle group (the whole point of the split, vs. an old 1×/week bro
  split).
- Emphasis on back/shoulders/arms/abs (his aesthetic priority) — no direct calf work, no direct
  glute work (both explicit choices; glutes get RDL/Bulgarian-split-squat carryover only).
- When reordering sessions, keep: Chest/Back sessions ~4 days apart where possible, Legs sessions
  ~4 days apart, Shoulders ~2 days apart. A single heavy barbell lift per day keeps sessions to
  ~60-75 min.
- Abs target ~20-22 sets/week across Lower A + Lower B (deliberately above generic research optimum
  — this is intentional, abs are priority #1).

## Venue / equipment comparability rules (IMPORTANT — read before judging any progression)

Yannick trains across Basic Fit gyms, a home garage gym (squat cage + pulley + Nuobell adjustable
dumbbells — a legitimate setup), hotels while traveling, and occasionally "Other."

- **Barbell and dumbbell weights are fully comparable across every venue.** Absolute load is
  physically identical anywhere — squat, RDL, bench, OHP, incline DB press, DB curls, etc.
- **Cable/pulley and machine weights are NOT comparable across venues.** Different resistance
  ratios, pulley mechanics, and calibration mean "45kg" at one gym ≠ "45kg" at another. There is
  **no reliable universal conversion ratio** between venues — the gap is exercise-specific (e.g. one
  exercise might read ~15% heavier at home vs Basic Fit, while another reads identical). Compare
  same-location only.
- **Basic Fit Martigues and Basic Fit Saint-Louis are tagged as ONE location, "Basic Fit"** (final
  decision, reverses an earlier attempt to keep them separate — splitting them broke the app's
  "last time at this location" reference lookup). Known nuance: **Martigues can *feel* heavier at
  the same logged weight than Saint-Louis** (pure perceived-effort/RPE difference) — don't read a
  harder Martigques session as a real strength regression if the logged weight matches history.
- **ToTheLimitGym is a separate location baseline.** Basic Fit Saint-Louis closed in late Sep 2026,
  and Yannick moved to ToTheLimitGym, whose equipment is nicer but mechanically different. Never
  compare its cable/machine loads directly with Basic Fit; build new same-location references.
  For the initial adaptation sessions, also avoid judging barbell performance as a regression:
  the bench, rack, bar and plates are all unfamiliar, even though nominal free-weight loads should
  become broadly comparable once Yannick is adapted to the setup.
  The gym's official equipment page describes it as a Hammer Strength performance center with
  140+ machines. Public photos confirm a large plate-loaded and selectorized leg area, but do not
  provide a reliable model-by-model inventory; use Yannick's machine labels/photos for exercise
  selection rather than identifying equipment from wide-angle marketing images.
- **OHP variant nuance**: home OHP is often done seated (low garage roof) vs standing at a gym.
  Seated is a stricter variant (no leg drive) — compare seated-to-seated, standing-to-standing, even
  though the barbell weight itself is comparable.
- The app (`yf-tracker`) has a **location picker** per session (Home / Basic Fit / ToTheLimitGym / Hotel / Other)
  built specifically to support this — every session should be tagged, and the training-log CSV
  note column carries a `[Location]` prefix from the merge script.
- **Bench press: read UNASSISTED-only from ~Sep 10 2026 onward.** He dropped the spotter around
  then — earlier "assisted" numbers are not comparable to current ones. Treat the switch to
  unassisted as a deliberate baseline reset, not a strength loss. See `00_START_HERE.md` for the
  current target.

## Bulgarian split squat — dumbbell convention (sciatic caution, live)

Default is **one dumbbell** (not two): (1) Yannick has a history of sciatic-like nerve pain with
two dumbbells (years ago) — never load through any nerve/radiating sensation, revert to one DB
immediately if it recurs; (2) one DB also emphasizes glutes more, useful since he does no direct
glute work. As of Aug 2026 he's been using 2 DB (e.g. 18kg+18kg) without pain — currently tolerated,
but the caution stays live. If ever logged with 2 DB, the weight is the TOTAL across both hands —
don't compare 1-DB vs 2-DB numbers directly, they're different difficulty per kg.

## ToTheLimitGym equipment inventory and redesign (provisional, Sep 29 2026)

The existing YF-UL5 v1.4 sessions remain the Basic Fit/home fallback. A ToTheLimitGym-specific
variant is being designed only after equipment testing; do not overwrite the stable program until
the exercise choices are confirmed.

First Lower B equipment test (loads are new-location baselines, not comparable to Basic Fit):
- Hack squat: liked; 20×12 @5, 30×10 @4, 40×8 @3, 50×5 @2.
- Linear leg press (empty carriage noted as 10): tested 20×10 and 40×10; Yannick was not a fan.
- Belt squat: liked; 40×10 and 40×8. Hole/lever setup matters; closest-to-machine setting felt
  best, with other tested positions noted as second/fourth hole.
- Hammer Strength leg extension: liked, similar feel to Basic Fit; 75×10, 82×10.
- Hammer Strength Iso-Lateral Kneeling Leg Curl: exceptional sensation; 10×20, 20×8.
- Hammer Strength leg curl: good, similar to Basic Fit; 46×9.
- Hammer Strength Glute Drive: excellent sensation; 10×12, 30×10 easy.
- Ab bench (exact model/name to confirm by photo): excellent sensation; 5×15, 10×12.
- Hammer Strength oblique crunch: excellent sensation; 5×22, 10×10.

Early direction, not yet final: likely replace back squat in the ToTheLimitGym variant with stable
machine work for hypertrophy; use hack squat and belt squat as distinct primary knee-dominant
slots across Lower A/B, retain RDL/hinge and unilateral coverage, and use the kneeling and regular
leg curls to improve hamstring frequency. Replace existing ab work rather than simply adding the
two new ab machines. Linear leg press is currently a poor candidate because Yannick disliked it.
Photos/model plates are still required before locking resistance-profile or redundancy decisions.

### YF-UL5 v1.5 — ToTheLimitGym variants (Sep 29 2026, program file imported via app)

`yf-tracker-program-v1.5.json` (iCloud folder) keeps all five v1.4 sessions byte-identical as the
Basic Fit/home fallback and adds three TTL sessions. Upper A and Upper B are shared (unchanged).
- **Lower A · TTL**: Belt squat 4×8-12 → RDL 4×8 → Glute Drive (HS) 3×8-12 → Leg curl allongé
  (HS) 3×10-12 → abs. (Glute Drive replaced Bulgarian split squat at Yannick's request, Sep 29 —
  this revises the old "no direct glute work" choice for the TTL variant; unilateral leg work is
  now fallback-only.)
- **Lower B · TTL**: Hack squat 4×8-12 → Kneeling leg curl iso-lateral (HS) 3×10-15/leg → Leg
  extension 3×10-15 → abs.
- **Upper C · TTL**: Iso-Lateral D.Y. Row (HS) 4×8-12 (lat slot; replaces the cable pulldown since
  Upper A already has two) → incline DB press → Iso-Lateral High Row (HS) 3×10-12 (upper-back/traps
  slot replacing Basic Fit's wide row; High Row existence at TTL to confirm, else HS Iso-Lateral
  Row with elbows flared) → DB flye → cable pullover → face pulls → lateral raises.
- **TTL abs (both lower days)**: Ab bench crunch 4×10-15, Oblique crunch machine (HS)
  3×12-15/side, Leg raise slow 2×12-15 — replaces jackknife/Russian twist/sit-ups/plank.
- Rationale: back squat removed at TTL (stable machine quads for hypertrophy; RDL keeps the heavy
  free-weight hinge); hamstring knee flexion now 2×/week instead of 1×.
- Deliberately NOT used: linear leg press (disliked, redundant), Iso-Lateral
  Decline Press (redundant with dips + bench), Iso-Lateral Chest/Back (combo press/pulldown;
  cable pulldown already has a TTL baseline), Iso-Lateral Bench Press (Upper A stays barbell;
  keep only as a no-spotter backup).
- Open: confirm the HS "regular" leg curl is lying (not seated); ab bench/oblique machine exact
  models; whether TTL has Iso-Lateral Shoulder Press (Upper B OHP question), High Row/Front
  Pulldown, or a pullover machine.

### YF-UL5 v1.6 — optional venue-specific Upper B exercise (Sep 30 2026)

- Added `Triceps Extension machine (HS) (optionnel)` for ToTheLimitGym: 3×8-12 @1 RIR.
- In the shared `Upper B` fallback it is an **alternative to Overhead triceps, not extra volume**:
  populate whichever exercise the venue supports and leave the other blank. Distinct names preserve
  independent performance histories when logs are analyzed. Yannick has since proposed a different
  TTL pairing; it has not been incorporated into an active program version.

### YF-UL5 v1.7 — Flame Sport lateral-raise alternative (Sep 30 2026)

- Added `Élévations latérales machine (Flame Sport 3PLX) (optionnel)` at 4×10-15 @1 RIR.
- It is a ToTheLimitGym alternative to conventional lateral raises, not automatic extra volume:
  keep both entries available but populate only the version performed.
- Manufacturer information confirms an independent dual-arm, plate-loaded machine with a very
  robust frame. Its fixed path and stability are strong practical hypertrophy features, but its
  simple lever geometry is not proven superior to cables and likely emphasizes the upper portion
  more than the lengthened bottom. Judge it primarily by comfort, target-muscle tension and clean
  load/rep progression.
- Rear delt cable fly remains the programmed Upper B exercise. ToTheLimitGym currently lacks a
  convenient cable setup, so Sep 30 used a rear-delt machine as a one-off substitution and its CSV
  rows were named separately. Do not add the machine to the app unless Yannick later decides to
  keep it; first look for a workable cable-fly setup at the gym.

### Oct 7 proposed TTL Upper B arrangement (not implemented)

- Yannick wants incline DB curl + rope triceps, preacher curl + Hammer Strength triceps extension,
  and reverse curl + hammer curl when training at TTL. The Oct 7 rope sets were exported under
  `Overhead triceps` with a `Triceps corde` note; their historical CSV label has not been changed.
- A separate `Upper B · TTL` session was drafted prematurely as v1.8, then its iCloud JSON was
  removed after Yannick raised the broader venue design issue. Do not treat v1.8 as active or
  prepare another program file before settling how venue-specific choices and loads should work.
- Desired design: one five-session program, with venue-aware exercise choices and working-load
  references. Yannick expects about 60% TTL, 20% Basic Fit, 10% home and 10% hotels.

### Oct 8 pending overall Upper B revision

Yannick plans to revise Upper B overall, not merely add a TTL-specific pairing. First finish the
app improvements, then agree the session's exercises, volume, pairings and venue variants. Keep
active program v1.7 and historical workouts unchanged in the meantime; the preview's current
Upper B arrangement is not the final training prescription. Do not create a new program export
or silently rewrite old Upper B sessions before that review.

## Biceps exercise selection (locked design, trains across the full strength curve)

1. **Incline DB curl** — stretched/long-head position, the "stretch" slot.
2. **Preacher curl** — contraction/peak position, the true complement to the incline curl (NOT
   another stretch movement). Distinct from the old "curl pupitre" naming — don't merge histories.
3. **Reverse curl EZ** — kept deliberately (brachioradialis/wrist extensors), NOT redundant with the
   other two even though it looks similar at a glance; this was explicitly reconsidered and kept.

**Preacher curl equipment (confirmed Oct 8):** Basic Fit uses a weight-stack machine; TTL uses
a plate-loaded Hammer Strength machine. They fill the same exercise slot but their logged loads
are not directly comparable. The plate-load convention (per arm versus total) is not confirmed.
Sep 30's 20x10 and Oct 7's 12.5x10 are both tagged TTL; the Basic Fit/TTL distinction alone does
not resolve which setup/load convention was used on those two dates. No workout data was changed.

## Program-change history (context only — the position is settled, see below)

- **v1.4 changes**: Upper C pull-ups → lat pulldown (hypertrophy consistency); Upper B biceps
  reworked to incline DB curl + preacher curl (both kept, reverse curl EZ retained).
- **A 4-day "Push/Legs + Pull/Legs" alternative was fully designed and evaluated (Sep 16 2026),
  then explicitly rejected** ("too much sacrifice" — it would drop traps, Bulgarian split squat, leg
  extension, cable pullover, unilateral pulldown, and cut side-delt frequency to 1×/week, in
  exchange for shorter ~42-50min sessions vs the current ~60-70min). **It is saved only as a
  time-crunch fallback** (if baby/work/travel makes the 5-day genuinely unsustainable) — do not
  re-propose switching to it as a standing option; that decision is closed.
- **Periodization was evaluated (Sep 23 2026) and judged unnecessary for now.** Plain double-
  progression (fill the rep range, then add load) is still producing steady gains on ~80% of his
  lifts (1-2.6% e1RM improvement per session). Only bench and incline DB press looked flat at the
  time, and both had just rebaselined/broken a stall rather than genuinely plateaued. Rule of thumb:
  only consider periodizing (e.g. a light daily-undulating scheme for bench: one heavy 4×4-6 session
  + one lighter-volume 3×8-12 session per week) after ~3-4 sessions of a genuine, unmoving stall —
  not before. Evidence basis: periodization gives a small strength edge for trained lifters at high
  frequency, but ~zero extra hypertrophy vs volume-equated linear progression — it's a strength
  fine-tuning tool, not the lever for his actual goal (adding muscle, which is a volume/surplus
  question, not a periodization one).

## e1RM formula used throughout analysis

```
e1RM = weight × (1 + reps / 30)      (Epley formula)
```

Used to compare "best set" across sessions even when rep counts differ.

## Strength-level context (established Sep 13 2026, still a useful anchor)

At ~73kg bodyweight, main lifts sit around 1.1-1.3× bodyweight (squat ~1.27×, RDL ~1.25×, bench
~1.14×, OHP ~0.62× — his weakest lift but also his fastest historical progressor). This reads as a
**legitimate low-to-mid intermediate lifter** — real trained muscle, but not a large absolute amount
yet. This is the basis for a recurring theme in his goals: getting leaner won't by itself produce
the "look like a lifter" physique he wants — that requires adding muscle (a bulk), then a final cut
to reveal it. See `04_profile_goals_and_style.md` for the full goal-priority framing.

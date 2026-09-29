# Recent Session Log (append-only narrative)

Chronological, oldest first. Add new dated entries at the bottom when something durable happens.
This is the "what actually happened recently" complement to the stable topic files (`00`-`06`) —
read it for continuity on decisions and tone, not as the source of current truth (that's
`00_START_HERE.md` plus the live CSVs).

## Sep 11 2026
Answered a direct factual question ("what was my last RDL best performance") from the training log:
most recent RDL was submaximal (70×8 @ RIR2-3), his real recent best was 71.5×8 (RIR1) three weeks
earlier — flagged as under-loading, recommended returning to 72.5kg.

## Sep 13 2026 — cut wind-down begins
Full weekly review: waist at an all-time low (80.0, tying a prior low), weight 7dMA barely moving
(−0.21kg/wk), strength holding — textbook recomp. But waist's rate of improvement had clearly
slowed (7dMA moved only 0.15cm over the fortnight) → **decision: start ramping the deficit down,
300→200**, framed explicitly as "the cut stopped paying, time to start feeding the muscle side"
rather than "stopping." Separately established (via a strength-vs-bodyweight analysis) that his
lifts sit at ~1.1-1.3× bodyweight — a real but not large intermediate level — meaning the
bottleneck to "looking like a lifter" is muscle, not remaining fat; cutting further would mostly
make him look skinnier, not more muscular. Also researched training-frequency evidence (Pelland
2024 meta-analysis: frequency doesn't matter when volume is equated) and, combined with a
basketball game eating into what should've been a rest day, agreed to trial a 4-day week for one
cycle (cycle 9) rather than the normal 5-day, purely as an in-cut recovery accommodation — not a
program change.

## Sep 14-17 2026 — cycle 9 (4-day trial), first ramp week
Deficit 200 confirmed active. Upper C session: incline DB press broke a month-long 24kg stall
(→26kg), wide-grip pulldown hit a rep-PR — initially mis-attributed to the (not-yet-started)
deficit change, self-corrected once the actual date math was checked (the PR session came off a
well-fed rest day, unrelated to the ramp). A Lower B/hack-squat session got mislabeled "Lower A" at
one point and was corrected from the actual exercise list in the log. Researched hack-squat machine
physics (empty sled weight + ~45° incline discount effective load) in response to a question about
what the number "means" — concluded machine numbers aren't standardizable across gyms/people, back
squat (~1.1× bodyweight) is the real strength gauge. A Thai-food dinner became a useful natural
experiment: comparing a heavy-soy-sauce version (Aug 27, big waist spike) against a light-soy-sauce
version of the same dish (Sep 17, tiny spike) established that **restaurant-meal waist water is
sodium/soy-dose-dependent**, distinct from the separate onion/garlic FODMAP-gas mechanism.

## Sep 18-19 2026
A 4-night cluster of ~4am wakes began (nightmare+pee / nightmare / pee / dry-thirst-wake), no
single consistent dietary trigger identified; broke on its own Sep 19. Weekly review (Sep 13-19):
ramp landing ~175-190 real deficit, both weight and waist still trending down, cluster broken.
Cycle 9 closed at 4/5 sessions (Lower A intentionally skipped per the recovery-trial design);
**cycle 10 opened Sep 19 with Upper A**, and the close-grip row hit its 52kg target with a clean PR
— confirming the earlier "under-loading" read (it wasn't a real plateau, he'd simply stopped
pushing the weight).

## Sep 20-23 2026
Sunday weekly review: squat hit 82.5×5 (new PR) and RDL hit 75×8 at RIR2 (a big jump past the
72.5kg target, since he clearly had more in the tank) — both while still in a deficit, a good sign
for the ramp. Deficit held at 200 (waist not stalled). Researched and discussed periodization —
concluded not needed yet, plain double-progression still working on most lifts. Waist crossed
**sub-80 for the first time** on Sep 21 (79.8) and confirmed real the next day (79.7) — the leanest
point of the entire cut. Around this point Yannick also reported a new visual observation: 4 abs
visible when contracted but barely when relaxed, alongside a feeling of "looking skinnier" even as
strength kept climbing. This was explained as two compatible-but-distinct things: (1) several
recent PRs were under-load *corrections*, not new hypertrophy; (2) a long deficit depletes muscle
glycogen, which flattens visual fullness reversibly, independent of any real muscle loss (strength
still rising is itself evidence muscle is being protected, not lost). A step to Deficit=100 was
recommended given the goal-2-vs-goal-1 tradeoff becoming visible — **Yannick explicitly declined
it**, preferring to re-decide fresh at the next clean weekly review rather than react immediately,
and reassured by the glycogen/strength explanation. Decision: **hold 200**, revisit "fresh on
Sunday" (Sep 27). A new minor, unconfirmed hypothesis was raised: dinner-time Greek yogurt possibly
contributing to a soft lower-belly look via overnight dairy digestion/bloat (not tested yet).

## Sep 24-26 2026 — south-of-France trip
Yannick got an **Apple Watch Series 12** (new high-frequency HR sensor, Readiness score, Sleep
Score). Researched the new hardware/features (post-training-cutoff) and gave watch-face/complication
recommendations tailored to his priorities (kept steps/exercise-shortcut/timer, added the new
Readiness-score and Heart-rate complications, kept Sleep Score given the active nocturia
investigation, merged weather into one slot). Separately declined a request to surface Apple's
"Total calories" (Active+Resting) as a watch complication — explained it isn't exposed as a native
complication anyway, and more importantly it's Apple's *own* TDEE estimate, the same number already
known to under-read his real deficit by ~200/day; using it would risk quietly reintroducing that
bias. A schedule change (rest day, then travel) meant Cycle 10 closed incomplete at 4/5 sessions
(missing its Lower B); **cycle 11 started Fri Sep 25 with Upper B instead of the usual Upper A**,
re-sequenced around the trip (checked and confirmed the muscle-group recovery spacing was actually
fine despite the unusual order). Weight tracking paused for the trip (no scale); waist-only
tracking Fri→Mon. Sessions during the trip: Upper B (Sep 25, Basic Fit, small PRs on OHP and
triceps pushdown) and Lower B (Sep 26, Basic Fit Martigues — hack squat felt heavier there, a known
Martigues venue-feel effect, not a real regression; jackknife progressed to 40kg). Sleep stayed
clean through the trip so far, no travel disruption. **iCloud sync repeatedly lagged during the
trip** (phone→iCloud upload delay) — worked around by having Yannick paste the exported JSON
directly into the conversation when needed, which was written straight to the iCloud path and
merged normally.

## Sep 27 2026 (status: unclear as of migration)
The planned "revisit 200-vs-100 fresh on Sunday" review does not appear to have happened in this
session before the migration to Codex began — **first thing a new session should do is check with
Yannick whether that review happened elsewhere, and if not, run it** using the waist trend from
`00_START_HERE.md` plus whatever fresh data exists by then.

## Sep 28 2026 — migration from Claude Code to Codex
Yannick decided to switch tools. This whole `docs/context/` folder plus the repo-root `AGENTS.md`
were created in this session specifically to carry everything forward: the memory files that used
to live automatically in Claude Code's memory system were read in full and reorganized into these
plain files so a new Codex session has full continuity without needing anything outside this repo.
No decisions were changed in the process — this was a context/format migration only.

## Sep 28 2026 — Sunday review deferred
Confirmed that the planned Sep 27 deficit review did not happen. While still in the south of
France, Yannick had no scale readings and skipped waist measurement for two days. He will resume
measurements Sep 29; for the Sep 28 Streamlit update, the last measured weight and waist are being
carried forward as reference values rather than treated as new observations. Review Deficit 200
vs 100 once the update is complete, interpreting the travel-period measurements accordingly.

## Sep 28 2026 — delayed deficit review
Ran the delayed review after the Streamlit update. Sep 21-27 nominal WADP deficit averaged only
~5 kcal/day because Sep 26 included 4,000 kcal consumed (a nominal 940 kcal surplus); excluding
that one-off day, the other six days averaged ~163 kcal/day deficit. Weekly waist average improved
from 80.21 to 79.80 cm and waist 14dMA reached 80.01 cm, but weight from Sep 25 onward and waist for
Sep 27-28 were carried forward during travel. The evidence is therefore insufficient for a clean
200-vs-100 decision. Hold Deficit=200 temporarily and reassess from fresh post-travel measurements
starting Sep 29.

Full weekly comparison (Sep 21-27 vs Sep 14-20): waist average 79.80 vs 80.21 cm and waist 14dMA
80.01 vs 80.26 cm; strength remained green across four sessions. Nominal WADP deficit averaged
5 kcal/day vs 177 prior week, entirely distorted by the one-off Sep 26 travel intake; the other
six days averaged 163 kcal/day. Sleep averaged 7h17 vs 7h21 and steps 8,494 vs 10,008. Weight
averages were not decision-grade because Sep 25-27 were carried forward. Sep 28 then delivered a
clean 8h24 recovery night with no wake or pee after Sep 27's short 5h03 night.

Training retro for Sep 21-27: four sessions and a clearly positive week overall. Sep 23 Upper C
was strongest: incline DB press 26×7 vs 26×6, wide row 52×8 vs 47×11, cable pullover 44×11 vs
40×12, with pulldown held at 66×9 and DB flye successfully moved 18→20kg. Sep 25 Basic Fit Upper B
produced standing OHP 37.5×7 vs 37.5×6, lateral raises 7×15 vs 6×15, overhead triceps 32×12 vs
32×10, preacher curl 18×12 vs 18×11, and pushdown 54×10 vs 50×12; reverse curl was the only small
dip (38×9 vs ×10). Sep 26 Lower B held hack squat at 50×8 despite the known heavier Martigues feel,
while leg curl and extension each added a rep and weighted jackknife moved 38→40kg at essentially
equal e1RM. No systemic regression or recovery warning.

## Sep 28 2026 — ToTheLimitGym and Upper A
Basic Fit Saint-Louis closed, so Yannick moved to ToTheLimitGym, which has nicer but different
equipment. Added it as a distinct location in yf-tracker app v1.6 and bumped the offline cache to
v6; its cable/machine loads must establish new baselines rather than compare with Basic Fit.
Merged Sep 28 Upper A from iCloud (one new session). The export was initially tagged Basic Fit;
after Yannick confirmed the venue, all 19 set rows were corrected to ToTheLimitGym. On
the first review, bench 67.5×4 @0 RIR was too readily compared with the prior Basic Fit 67.5×6 @1.
Yannick correctly noted that all equipment is different, including the bench setup, bar and
plates. Correction: treat the full Sep 28 session as the initial ToTheLimitGym baseline and do not
classify bench as a regression until same-location data exist.

## Sep 29 2026 — new-gym program reassessment begins
Confirmed the Sep 28 Upper A is stored as ToTheLimitGym on all 19 rows, and Yannick confirmed the
deployed v1.6 phone app is current. ToTheLimitGym has substantially better-looking leg equipment,
raising a possible future change from barbell squat toward machine-based quad work. No change was
made yet: use the first Lower B and Lower A visits to establish exact machine variants, setup and
feel, then reassess the two lower sessions while preserving a knee-dominant compound, hip hinge,
unilateral work and knee-flexion/extension coverage.

Yannick plans to take a one-year ToTheLimitGym subscription and gather manufacturer/model
references for the gym's machines over the following days. Once available, use the inventory and
his actual setup/feel notes to evaluate movement overlap and make a targeted program revision.

Web research found ToTheLimitGym's official equipment page: it markets the facility as a Hammer
Strength performance center with 140+ machines, and its leg-area photo confirms several
plate-loaded stations plus selectorized leg equipment. The public page does not list exact models,
so the program redesign still needs Yannick's machine references and hands-on feedback. Sep 29
Lower B was completed, but its new export had not yet reached iCloud even after a sync nudge.

Yannick pasted the Sep 29 equipment-test notes directly. Clear positives were the hack squat, belt
squat, Hammer Strength Iso-Lateral Kneeling Leg Curl, Glute Drive, ab bench and oblique crunch;
leg extension and regular leg curl were also good, while the linear leg press was not liked. The
working direction is a ToTheLimitGym-specific hypertrophy variant that may remove back squats,
while retaining YF-UL5 v1.4 as the Basic Fit/home fallback. No final program revision yet: exact
machine photos/models and upper-body trials are still needed. The pasted sets were not manually
written to training_log.csv because several RIR values were incomplete and the export may arrive.

## Sep 29 2026 — measurements resume
First real post-travel measurement: 72.7 kg and 79.9 cm waist after 71.9/79.7 had been carried
forward. The scale rebound with almost unchanged waist is consistent with restored food,
glycogen and water rather than fat gain. Sleep duration was 6h39; wake/pee quality still pending.
Hold Deficit=200 and wait for several fresh mornings before revisiting 200 vs 100.

Sep 29 sleep detail: no pee, but sleep onset was difficult; 6h39 total. For Budapest Oct 3-5,
Yannick expects up to two poor nights and less controlled food. If both nights are bad, he plans to
book a separate hotel for Sunday-to-Monday. Agreed to treat this as travel noise and not compensate
with extra restriction or forced training.

## Sep 29 2026 — YF-UL5 v1.5 (ToTheLimitGym variants)
From Yannick's equipment test notes plus Hammer Strength model names (Iso-Lateral D.Y. Row,
Chest/Back, Decline Press, Bench Press), built v1.5: three new TTL sessions (Lower A · TTL,
Lower B · TTL, Upper C · TTL), all five v1.4 sessions kept unchanged. Kept exercise names where
the movement is the same (Hack squat, Leg extension, Leg curl allongé, RDL, BSS) so the app's
per-location lookup separates TTL baselines automatically. Glute Drive excluded on purpose
(standing no-direct-glute choice). Program JSON written to iCloud for import; app label bumped to
program v1.5 and sw cache v7 in yf-tracker (uncommitted — Yannick commits/pushes).

Revision same day (Yannick's feedback): Lower A · TTL swaps Bulgarian split squat → Glute Drive
(he didn't recall the no-direct-glute rule and prefers it). Upper C · TTL reworked to machine
equivalents of the same muscles: D.Y. Row takes the lat slot (Upper A keeps the cable pulldowns),
Hammer Strength Iso-Lateral High Row takes the wide-row upper-back/traps slot (to confirm it
exists at TTL).

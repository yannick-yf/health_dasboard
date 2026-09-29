# Profile, Goals, and Working Style

## Who Yannick is

- 34 years old (born 1992), male, 185cm, based in Alsace, France (Saint-Louis/Basel area).
- Office job with a ~30min bike commute each way (counts toward daily NEAT/TDEE).
- Wife Mounia; first child Malya born **December 23, 2025**.
- Plays basketball occasionally (real conditioning load, not "light" activity — a 2-hour session
  is a legitimate training day, factor it into recovery/scheduling, not just calories).
- Has a home garage gym: squat cage + pulley system + Nuobell adjustable dumbbells (a legitimate
  A-grade setup for barbell/dumbbell work; the pulley is B-grade vs a commercial gym cable — see
  `02_training_program.md` venue rules).
- Trains at Basic Fit (multiple locations, treated as one for tracking purposes) and occasionally
  hotel gyms while traveling.

## Newborn context — PERMANENT, read before judging any post-Dec-23-2025 data

Chronic sleep restriction from the newborn is a **structural constraint, not a controllable
variable or a lever to push through.** Concretely:
- Realistic sleep window is 5.5-7h/night with frequent fragmentation — don't flag sub-8h sleep as
  a failure or a missed target.
- Expect roughly: lower muscle protein synthesis, somewhat lower testosterone, elevated evening
  cortisol, and preferential abdominal fat storage, all as physiological consequences of this phase
  of life — not behavioral shortcomings.
- **When designing a cut, prefer lengthening the window (more weeks, smaller daily deficit) over
  steepening it** — better muscle retention under sleep restriction. This principle has been
  applied consistently (e.g. the current cut's slow ramp-down rather than an abrupt stop).
- Adherence dips after a bad-sleep night are predictable consequences, not failures to correct.

## Goals, in priority order (as stated by Yannick, current)

1. **Athletic body with 4 abs visible in NORMAL light** (not just flexed/pumped) — the literal
   visibility target.
2. **A physique that clearly shows he lifts** — his wife's phrase was that he currently looks more
   "marathonian" than "shredded lifter." This is explicitly a **muscle** problem, not a fat problem
   — established analytically in Sep 2026 (his lifts sit around 1.1-1.3× bodyweight, a legitimate
   but not large intermediate level). Getting leaner alone will not solve this; it needs added
   muscle mass via a bulk, then a final reveal-cut.
2. **Building lean muscle** generally.
3. Longer-term: cut → clean bulk cycling, not a single permanent state.

**Known tension between goal 1 and goal 2**: pushing the deficit further chases goal 1 (visible
abs) while actively working against goal 2 (looking bigger/more muscular) — getting leaner on an
under-muscled frame makes him look *more* like a skinny distance-runner, not less. This tension has
been named explicitly and is the reason the cut is being wound down via a slow deficit ramp rather
than pushed further, even though waist is still improving. Don't resolve this tension by defaulting
to "just cut more" — check which goal he's currently prioritizing before recommending more deficit.

## Protein — tracked externally, not in the CSV

Floor of **≥160g/day**, which he typically reaches or exceeds. This is above the muscle-protein-
synthesis ceiling for his bodyweight (~1.6-2.0g/kg, Morton 2018 meta-analysis) — adequate for both
hypertrophy and muscle preservation in a deficit. **Do not flag protein as unknown/under-tracked in
any analysis**, and do not attribute any bulk/cut outcome to protein inadequacy — if partitioning
looks off, the cause is elsewhere (training volume, sleep, deficit/surplus calibration).

## Working style / communication preferences (established over many sessions)

- **English only.**
- **Direct, honest feedback — willing to disagree, no accommodating a bad idea just to be
  agreeable.** He values being told plainly when an idea is a poor trade-off (e.g. the 4-day
  program redesign was told clearly it involved "too much sacrifice" rather than being softened).
- **Data over theory, always, in any conflict.** If a formula/model predicts X and the CSV shows Y,
  trust the CSV — recalibrate the theory, not the observed data.
- **He actively cross-checks your reasoning and calls out overreach** — e.g. attributing a data
  movement to the wrong cause, or reading too much into a single day's number instead of a trend.
  When he pushes back, actually redo the analysis/math rather than just rephrasing the same
  conclusion more gently. This has happened multiple times and each time the correction was
  substantive (wrong attribution of a waist change, premature "you're near the goal" framing,
  crediting a nutrition change that hadn't started yet, etc.) — treat his pushback as a signal to
  genuinely re-derive, not just placate.
- **Prefers a clear recommendation over an exhaustive menu of options**, unless he explicitly asks
  to compare alternatives (in which case a structured comparison is appropriate).
- **Doesn't want recurring automation pitched** (`/schedule`-style recurring jobs, cron-style
  agents) — he already declined this once on cost/preference grounds; he prefers to come back and
  ask for analysis on demand. Isolated one-off scheduled follow-ups (e.g. "check back on X in 3
  weeks") are still fine to offer if genuinely useful.
- When a topic depends on information past a knowledge cutoff (new hardware, new research, new app
  features) — **search rather than guess**, and cite sources.
- Never send his email or any personal data to an unrelated third-party service.
- Confirm before writing to CSV outside the routine daily merge/sleep-log flow; never auto-commit
  data changes to git — that's his call.

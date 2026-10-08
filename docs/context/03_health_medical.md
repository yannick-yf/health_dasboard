# Health & Medical Context

Read before any nutrition, sleep, or supplement discussion. For live/open items see
`00_START_HERE.md`; this file is the stable reference detail underneath it.

## Gilbert's syndrome (likely) — PERMANENT context, benign

Found in December 2025 bloodwork: elevated unconjugated bilirubin (18.9 µmol/L, ref <17.1) with all
liver enzymes completely normal (ASAT 31, ALAT 36, GGT 20), lipase normal, no hemolysis. This
pattern is classic Gilbert's syndrome (3-10% population prevalence) — benign, no treatment needed,
not yet formally confirmed by a GP visit. Known triggers that worsen it (his life currently hits
most of them): fasting/skipping meals, intense daily exercise, dehydration, poor sleep, alcohol.
**Implication: occasional yellow-tinge stool episodes are diet/transit related, NOT alarming, and
not caused by Gilbert's directly** (unconjugated bilirubin doesn't reach the intestine in excess).
Persistent yellow AND greasy/floating stool would be the actual flag (steatorrhea workup) — not
seen so far.

Other Dec 2025 bloodwork notes: mild microcytosis flag (VGM 80fL low-normal, TCMH 26.0 low) →
ferritin/iron/transferrin saturation should be checked next panel. Folate low-end-normal. Vitamin
B12, TSH, CRP, lipid profile all fine. Testosterone 21.43 nmol/L (6.17 ng/mL) — solid, not a
concern (relevant context if a testosterone-boosting supplement ever comes up again — see below).

## FODMAP sensitivity — active, ongoing personal tolerance mapping

Chronic daily gas + occasional yellow stools were traced (Jun 2026) primarily to **Wasa rye
crispbread** (10 slices/day, ~10-15× low-FODMAP threshold) — permanently removed, replaced by
120g raw basmati rice at breakfast. Produced ~90% symptom resolution within a week; case closed,
no reintroduction planned.

**Confirmed/suspected trigger list (avoid or dose-limit):**
- **Rye** (Wasa) — confirmed primary trigger, permanently removed.
- **Onion/garlic, especially cooked into sauces** (e.g. Indian, Thai restaurant food) — confirmed
  repeat trigger, causes GI gas/upset AND water-retention waist spikes. Cooking does NOT remove the
  fructans; they leach into the dish's liquid and get eaten anyway.
- **Mushrooms** (mannitol polyol) — suspected trigger, notably at Basic Fit-adjacent restaurant
  meals; watch for confirmation on repeat exposure.
- Apple, honey >10g/serving, fresh dairy in volume, wheat/rye-heavy products — presumed problematic
  until proven otherwise for any new food.
- **Dates** (a fructan source) — flagged Sep 2026 as a common ingredient across several "healthy"
  protein-food products he's testing (see below); not yet confirmed personally, but a known
  category risk given the rye/onion pattern.
- **Legumes / red-kidney-bean flour** — same GOS-risk category, one product under test contains it.
- **Cashews** — high-FODMAP nut, present in one product under test.

**Tolerated / safe:** basmati rice (any serving size), Greek yogurt 0% (~150g, low lactose),
lactase-treated milk (Lactel Matin Léger), blueberries ≤40g, dark chocolate 85% ≤30g, honey at his
normal ~10g/day baseline.

**Key non-obvious finding for reading his data**: "date-sweetened / no added sugar" packaged foods
are often a FODMAP red flag, not a green one — dates are high-FODMAP (fructans) while ordinary table
sugar is low-FODMAP. When evaluating a new snack/food product for him, scan the ingredient list for
**dates, legume flours, cashews, chicory/inulin fiber** as the actual risk flags — not sugar content.

**Active personal experiment (Sep 2026):** he bought 3 Nutripure products (a buckwheat-based
granola, a protein bar, a bean-based chocolate spread) specifically to test tolerance one at a
time, watching gas/stool for 24-48h after each. Ask for results if not yet reported. Rough
prediction going in: granola (buckwheat base, gluten-free, dates only) is the best-case fit; the
bean-based spread (legume + dates, double FODMAP hit) is the highest-risk of the three.

**Sodium vs FODMAP — a distinction that matters for reading his waist data.** A restaurant-meal
waist spike can come from two independent mechanisms and they should not be conflated:
- **Sodium (e.g. soy sauce, salty sauces) → water retention → waist bump**, and this is clearly
  **dose-dependent** (proven via a controlled comparison: same Thai dish with heavy soy sauce
  produced a much bigger waist spike than the same dish with light soy sauce).
- **Onion/garlic FODMAP → gas/GI distension**, a separate mechanism, doesn't reliably track sodium
  dose the same way.
When a restaurant meal causes a waist bump, ask what was actually salty/saucy vs what was
onion/garlic-heavy before attributing it to either mechanism.

## Nocturia investigation — mostly resolved as behavioral, low-grade open thread

Started Jul 2026 after Yannick reported frequently waking ~4am to urinate. Tracked in
`data/sleep_log.csv` (see `05_app_and_data_pipeline.md` for schema). The `urge` field distinguishes
bladder-driven (nocturia proper) from awake-anyway (external wake, e.g. baby, nightmare, then pees
opportunistically) — this distinction matters, don't count awake-anyway nights as nocturia.

**Pattern established over ~2 months of tracking:** overwhelmingly behavioral/dietary — clean
nights are common when he's home, eating normal low-sodium/low-FODMAP food, no alcohol. Triggers
that reliably produce a pee-wake: alcohol + heavy salty/fatty dinner combo, big pre-bed fluid loads,
active cut-related water release (whooshing). A clean ~7pm fluid cutoff produced his best-ever
night early in the investigation — worth suggesting again if wakes recur.

**A 4-night ~4am wake cluster occurred Sep 15-18 2026** (nightmare+pee / nightmare / pee / dry-
thirst-wake) with no single common dietary trigger, then broke Sep 19-22, then recurred once more
Sep 23. Yannick reports he's not stressed. Leading hypotheses, in order: (1) the cut itself — a long
sustained deficit is a well-documented cause of early-morning waking; (2) dry indoor air as heating
season starts in Alsace. Both are cheap to test (ease the deficit — already happening; try a
humidifier / cracked window for a few nights). Not yet conclusively resolved.

**Oct 6→7 sleep:** 8h50 recorded in the health row, but fragmented: woke around 02:00, could not
sleep until around 04:00, then slept until 08:00–09:00. The reason for waking, including whether
it was bladder-driven, has not been established; do not classify it as nocturia without that detail.

**Medical backstop, low-urgency but should not be dropped:** because thirst + nocturia together are
a classic (if usually benign) early sign to rule out, add **fasting glucose + HbA1c** to the next
bloodwork panel. His Dec 2025 panel didn't include these. This has been agreed with Yannick as a
"when he next gets bloodwork" item, not an urgent one.

**Standing instruction: ask about the previous night every daily data-update, if he doesn't
volunteer it, and append one line to `data/sleep_log.csv`.**

## Supplements

- **Creatine (Nutripure, Creapure form)** — planned, not confirmed started as of last check. When
  he starts, expect a **+1-1.5kg water-weight bump over 1-2 weeks** that must NOT be misread as fat
  gain on the scale — ask for the start date and tag it explicitly when it happens.
- **Magnesium Taurine B6** — being considered, plausible given his sleep-quality focus, not decided.
- **Ashwagandha, L-Citrulline, Collagène Marin** — discussed once, judged optional/low-priority
  (ashwagandha genuinely has modest evidence for stress/cortisol/sleep in stressed men and could
  suit the newborn-stress period; the others are lower value for his goals).
- **A branded "Testosterone Booster" product (Tribulus, Maca, Ashwagandha, Zinc) was evaluated and
  declined.** His testosterone is already normal (21.43 nmol/L, Dec 2025) — most of these
  ingredients only move the needle in deficient men, so it would do essentially nothing for him.
  The one relevant lever for testosterone is that a long calorie deficit itself can suppress it —
  i.e. easing the cut (already the plan) is the real "testosterone support" action, not a pill.
- If ashwagandha alone is ever revisited: prefer a standalone standardized extract (KSM66 or
  Shoden, 300-600mg) over a blended product, partly to control dose and partly because ashwagandha
  has rare but documented liver-injury case reports — worth being a little more careful given the
  Gilbert's context (though Gilbert's itself doesn't raise this risk directly, it's the reason his
  liver markers get more attention than average).

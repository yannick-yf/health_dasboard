# Health Dashboard — Claude Code Context

## About This Project
Personal health tracking & analysis system. Goal: streamline weekly reporting, improve data-driven insights, and leverage AI for research-backed recommendations around a clean bulk program.

## Personal Profile
- **Name**: Yannick, 33y male, 185cm
- **Body fat**: ~13% (as of project start)
- **Program**: Clean bulk (2-3 month intermediate muscle building cycle)
- **Reference docs**: `docs/` — 3 program PDFs + NEAT/metabolism research papers
- **Data file**: `data/health_data.csv` (daily tracking from Aug 2025, ~7 months)
- **Personal info**: `data/personal_info.json`

## Current Goals (in priority order)
1. Streamline weekly report generation — automate analysis from CSV
2. Improve data pipeline — easier weekly data entry
3. Transition frontend: Streamlit → something better (candidate: Marimo)
4. Enrich AI agent with up-to-date research (currently RAG from PDFs)
5. Preserve clean bulk program history & context

## Data Schema (`data/health_data.csv`)
```
date, steps, sleep_min, workout_duration_min_tot, weight(kg),
calories_burned, calories_consumed
```
- Date range: 2025-08-04 → present (update weekly)
- Weight is sometimes imputed (look for `.0` precision vs decimals as imputation signal)
- `calories_consumed` = tracked intake; `calories_burned` = TDEE estimate

## Project Structure
```
health_dasboard/
├── CLAUDE.md                   ← you are here
├── data/
│   ├── health_data.csv         ← primary data source (update weekly)
│   ├── personal_info.json      ← personal metrics
│   └── sample_daily_backup.csv ← backup reference
├── docs/
│   ├── *.pdf                   ← bulk program protocols (3 PDFs)
│   └── research_papers/        ← NEAT, metabolism papers
├── fitness_agent/
│   ├── fitness_agent_rag.py    ← RAG agent (Agno + LanceDB + OpenAI)
│   ├── fitness_agent_simple.py ← simple agent (no RAG)
│   └── rag_docs/               ← PDF source for RAG knowledge base
├── frontend/
│   ├── app.py                  ← Streamlit entry point (being replaced)
│   ├── sections/               ← dashboard sections
│   └── utils/                  ← helpers
├── notebooks/                  ← analysis notebooks
└── .claude/commands/           ← project-specific slash commands
```

## Tech Stack
- **Runtime**: Python 3.11, managed with Poetry
- **Current frontend**: Streamlit (being transitioned)
- **AI agent**: Agno 2.0 + OpenAI + LanceDB (RAG)
- **Data**: pandas, plotly, scipy, numpy
- **Dev tools**: black (line=100), pytest, pylint, mypy

## Key Formulas Used
- **BMR**: Mifflin-St Jeor
- **TDEE**: BMR × activity multiplier
- **Energy balance**: calories_consumed - calories_burned

## Development Commands
```bash
poetry run poe run       # start Streamlit app
poetry run poe test      # run tests
poetry run poe format    # black format
poetry run poe test-cov  # coverage report
```

## Workflow Preferences
- Keep data updates simple: append rows to `data/health_data.csv`
- Weekly review on Sundays — generate 7-day summary + trend analysis
- Preserve historical data — never overwrite, only append
- Use research PDFs as ground truth for program adjustments

## Context from Previous Work (claude.ai project)
- Program started as conservative lean bulk, evolved to clean bulk
- Historical weight trend: ~74kg start → ~71-72kg mid-program → currently ~74kg (bulk phase)
- Key insight: NEAT variability is high (steps 6k-23k/day) — factor into TDEE estimates
- Sleep consistency is a tracked priority (target: 480min/8h)
- RAG knowledge base already loaded with program PDFs + research papers

## What NOT to do
- Don't auto-commit data changes — always review first
- Don't modify `data/health_data.csv` headers
- Don't add new dependencies without checking pyproject.toml first
- Don't replace existing analysis logic without reading the current implementation

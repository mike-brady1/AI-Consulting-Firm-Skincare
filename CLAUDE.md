# CLAUDE.md — AI Consulting Firm (Skincare)

Context for Claude Code sessions on this repo. Read before changing anything.

## What this project is

MSc 2 group project (generative AI / multi-LLM course, ECE Paris). The system acts as a **digital consulting firm**
for a skincare / dermatology-cosmetics brand:

1. The client company's **objectives** are the anchor (turned into measurable KPIs).
2. Several LLMs **generate different strategic scenarios**.
3. Each scenario is **tested on a simulated client base** (persona agents).
4. An **auditor** scores scenarios against the KPIs and EU cosmetics rules.
5. The system **picks the best scenario or merges several** into a better one, and shows ranked
   recommendations in a "Mini Netflix" view.

Background decision: the team first wanted to model the processes of large consulting firms, but there was not
enough data. The objective-anchored, simulate-and-audit approach replaced it. Don't reintroduce "imitate McKinsey".

Current providers (`.env.example`): Anthropic (orchestrator + generator), OpenAI (generator, personas, baseline),
Google Gemini (generator), Mistral (auditor). One API key per provider; the README lists them.

Architecture diagram (Whiteboard): https://claude.ai/artifact/VgJqsrKfx9WbgWsxbmCYca

## Architecture and roles

| Component | Kind | Notes |
|---|---|---|
| Orchestrator ("MT") | LLM | Objective → KPIs, detects vague/contradictory goals, refuses forbidden requests, lists requested tool actions, runs the loop |
| Scenario generators | Multi-LLM | 2–3 different providers, each with a consulting style (growth, cost, brand) — diversity is the point |
| Client simulators | LLM | Persona agents by skin type, concern, age, budget, channel. Cheap model, many calls |
| Synthetic client base | Gen AI | Personas must be grounded in real/authorised market data, otherwise they stay "encore hypothétique" |
| Visual creatives | **Diffusion** | Ad visuals / packaging per scenario (SDXL-Turbo, Flux). Required: the official brief demands a diffusion experiment |
| Auditor | LLM | **Must be a different model from the generators** (no model grades its own work; `pipeline.py` refuses to start otherwise). KPI score + compliance |
| Model bandit | Classic algorithm | Thompson sampling / UCB to allocate simulation budget to promising scenarios. Not an LLM |
| Merger | LLM | Combines top scenarios, must check the result is still coherent |
| Mini Netflix | UI | Ranked scenario cards: *comment* (how), *qui* (who), *chiffre* (figures). Crown = winner |

## Repo layout

- `src/pipeline.py` — candidate chain (orchestrator → generators → personas → auditor → best) and single-LLM baseline
- `src/prompts.py` — all prompts; **bump `PROMPT_VERSION` on every prompt change**
- `src/llm.py` — LiteLLM wrapper, JSONL tracing to `experiments/traces/`, `MOCK=1` offline mode.
  `seed` is only sent to providers that accept it (not Anthropic) and `temperature` only to models that accept it
  (not the Claude 5.5 family); traces record `seed_applied`, `temperature_applied` and `model_served`
- `src/tools.py` — permission gate; `publish_ad` is always refused (human approval required)
- `src/run_eval.py` — runs baseline vs candidate on `tests/cases.json` → `experiments/results.csv`
- `src/checks.py` — automated per-case checks mirroring each case's `verification` text (`case_pass` column).
  A new test case needs an entry in `CASE_CHECKS`; without one only the refusal decision is checked
- `src/diffusion_experiment.py` — 4 visuals × (baseline 1 step vs candidate 4 steps), fixed seeds, Fitzpatrick skin-tone check
- `tests/cases.json` — 10 frozen test cases (normal, limit, error, refusal, refused tool action). **Don't edit frozen cases**; add new ones with new ids
- `data/personas.json` — synthetic personas
- `docs/03_preuve_technique.md` — C3 report template (French)

Commands:
```bash
pip install -r requirements.txt
cp .env.example .env               # add API keys; never commit .env
MOCK=1 python src/run_eval.py      # plumbing test only
python src/run_eval.py             # real run
python src/diffusion_experiment.py --model stabilityai/sdxl-turbo   # needs a GPU (Colab T4)
```

## Rules (from the course brief and team decisions)

- **Never commit secrets, unauthorised documents or raw client data.** Personas and client data are synthetic.
- **Label every reported result** as *mesuré*, *calculé* or *encore hypothétique*. Mock-mode numbers are never results.
- Simulated persona reactions are simulations, not market evidence — say so wherever they are reported.
- The LLM auditor is a biased judge: keep a human check on a few outputs and report the agreement.
- Every experiment must be reproducible: log model, prompt version, temperature, seed (traces already do this).
- Decisions in the docs must follow from results, not impressions.
- Skincare compliance: no therapeutic claims ("cures", "treats" a disease), no fake reviews, no fake before/after images
  (EU Regulation 1223/2009, claims criteria 655/2013).
- Diffusion bias: test generations across Fitzpatrick skin types and document the outcome.
- Course deliverables (docs/, README, PR text) are written in **French**; code and comments in English.
- Work on branches and open pull requests for teammate review; don't push straight to `main`.

## C3 milestone (first GenAI proof) — required evidence

- 8–10 frozen test cases with expected result and verification mode → `tests/cases.json` ✅
- Minimal working generation chain (no full UI) → `src/pipeline.py` ✅ (only tested in mock mode so far)
- Baseline tested on the same cases + at least 1 explicit metric → `experiments/results.csv` ⏳ needs a real run
- 1 reproducible incident + 1 fix hypothesis (likely candidate: diffusion skin-tone bias) ⏳
- README updated with paths to evidence and task owners ⏳ owners still to fill
- Diffusion: 4 generations with seed, settings, time per output, quality comment ⏳

## Next steps

1. Real run of `run_eval.py` with API keys; fill the table in `docs/03_preuve_technique.md`.
2. Diffusion run on GPU; annotate `skin_tone_ok_manual` and `quality_note` in `experiments/diffusion_results.csv`.
3. Human review of 3 auditor outputs.
4. Replace the simple `combined` score in `run_candidate` with a bandit (Thompson sampling) and add the merge step.
5. Feed diffusion visuals to vision-capable persona agents.
6. Build the Mini Netflix interface.

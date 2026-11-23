# AI Consulting Firm — Skincare

Cabinet de conseil IA multi-LLM pour une marque de skincare : à partir des objectifs du client, le système
génère plusieurs scénarios stratégiques, les teste sur une base de clients simulés, les audite par rapport
aux KPI et recommande le meilleur scénario (ou une fusion de plusieurs).

**Dépôt :** https://github.com/mike-brady1/AI-Consulting-Firm-Skincare

## Architecture

Schéma complet (Whiteboard) : https://claude.ai/artifact/VgJqsrKfx9WbgWsxbmCYca — contexte pour Claude Code : `CLAUDE.md`.

```
Objectifs client ─► Orchestrateur (MT) ─► Générateurs (multi-LLM) ─► Visuels (diffusion)
                                                                          │
              Base clients synthétique ─► Personas simulés ◄──────────────┘
                                                │ réactions
Mini Netflix ◄─ Fusion ◄─ Bandit (Thompson/UCB) ◄─ Auditeur (LLM différent : KPI + conformité UE)
```

| Rôle | Type | Statut C3 |
|---|---|---|
| Orchestrateur | LLM | implémenté (`src/pipeline.py`) |
| Générateurs de scénarios (2–3 LLM, styles différents) | Multi-LLM | implémenté |
| Personas clients simulés | LLM | implémenté (3 personas, `data/personas.json`) |
| Auditeur (KPI + allégations cosmétiques UE) | LLM | implémenté |
| Outil `publish_ad` (validation humaine obligatoire) | Agent / outil | implémenté (refus systématique) |
| Visuels publicitaires | Diffusion | script d'expérience (`src/diffusion_experiment.py`) |
| Bandit, fusion de scénarios, interface « Mini Netflix » | — | prochaines séances |

## Lancer la preuve technique (C3)

```bash
pip install -r requirements.txt
cp .env.example .env          # puis renseigner les clés API (ne jamais commiter .env)

MOCK=1 python src/run_eval.py # test de plomberie hors ligne : résultats NON valides
python src/run_eval.py        # vraie exécution baseline vs candidat sur tests/cases.json

python src/diffusion_experiment.py --model stabilityai/sdxl-turbo   # GPU recommandé (Colab T4)
```

## Où sont les preuves

| Livrable demandé | Fichier |
|---|---|
| Cas de test gelés (10) | `tests/cases.json` |
| Résultats baseline vs candidat | `experiments/results.csv` |
| Vérifications automatiques par cas (colonnes `case_pass`, `checks`) | `src/checks.py` |
| Résultats diffusion (seed, steps, temps, mémoire) | `experiments/diffusion_results.csv`, images dans `experiments/diffusion/` |
| Traces (modèle, prompt, version, seed, latence, coût) | `experiments/traces/<run_id>.jsonl` |
| Sorties complètes | `experiments/outputs/<run_id>.json` |
| Document de preuve technique | `docs/03_preuve_technique.md` |

## Données et sécurité

- Personas et base clients : **synthétiques**, aucune donnée client réelle.
- Aucune clé API dans le dépôt (`.env` est ignoré par git).
- Chaque résultat est marqué « mesuré », « calculé » ou « encore hypothétique ».

## Équipe et tâches de la prochaine séance

| Tâche | Responsable |
|---|---|
| Exécuter `run_eval.py` avec les vraies clés et remplir le tableau du doc C3 | _à compléter_ |
| Exécuter l'expérience diffusion et annoter les images (teinte de peau, qualité) | _à compléter_ |
| Relecture humaine de 3 sorties de l'auditeur | _à compléter_ |
| Bandit (Thompson sampling) + fusion de scénarios | _à compléter_ |
| Interface « Mini Netflix » | _à compléter_ |

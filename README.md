# AI Consulting Firm — Skincare

Cabinet de conseil IA multi-LLM pour une marque de skincare : à partir des objectifs du client, le système
génère plusieurs scénarios stratégiques, les teste sur une base de clients simulés, les audite par rapport
aux KPI et recommande le meilleur scénario (ou une fusion de plusieurs).

**Dépôt :** https://github.com/mike-brady1/AI-Consulting-Firm-Skincare

## Architecture

Schéma complet (Whiteboard) : https://claude.ai/artifact/VgJqsrKfx9WbgWsxbmCYca — contexte pour Claude Code : `CLAUDE.md`.

```
Objectifs client ─► Orchestrateur (MT) ─► Générateurs (multi-LLM) ─► Auditeur (LLM différent : KPI + conformité UE)
                                                                                 │ scénarios conformes
          Base clients synthétique (30) ─► Personas simulés ◄── Bandit (Thompson) ◄─┘
                                                │ réactions
                    Mini Netflix ◄─ Fusion ◄─ Meilleur scénario          Visuels (diffusion) : expérience à part
```

| Rôle | Type | Statut C3 |
|---|---|---|
| Orchestrateur | LLM | implémenté (`src/pipeline.py`) |
| Générateurs de scénarios (2–3 LLM, styles différents) | Multi-LLM | implémenté |
| Personas clients simulés | LLM | implémenté (30 personas, phototypes I à VI équilibrés, `data/personas.json`) |
| Auditeur (KPI + allégations cosmétiques UE) | LLM | implémenté |
| Outil `publish_ad` (validation humaine obligatoire) | Agent / outil | implémenté (refus systématique) |
| Visuels publicitaires | Diffusion | script d'expérience (`src/diffusion_experiment.py`) |
| Bandit (Thompson sampling « top-two » + arrêt anticipé) | Algorithme classique | implémenté (`src/bandit.py`) |
| Fusion de scénarios, interface « Mini Netflix » | — | prochaines séances |

## Lancer la preuve technique (C3)

```bash
pip install -r requirements.txt
cp .env.example .env          # puis renseigner les clés API (ne jamais commiter .env)

MOCK=1 python src/run_eval.py # test de plomberie hors ligne : résultats NON valides
python src/run_eval.py        # vraie exécution baseline vs candidat sur tests/cases.json

python src/diffusion_experiment.py --model stabilityai/sdxl-turbo   # GPU recommandé (Colab T4)
```

## Bandit : tester moins de clients, trouver le même gagnant

Chaque scénario conforme est un « bras » ; un tirage = un client simulé qui lit le scénario
(récompense = (score KPI de l'auditeur + intention d'achat) / 20). Le bandit (`src/bandit.py`) teste chaque
scénario sur 3 clients, puis envoie les clients suivants aux scénarios les plus prometteurs (Thompson sampling
« top-two ») et s'arrête dès que le gagnant est probable à 95 %, ou au bout de `PERSONA_BUDGET` appels (30 par défaut,
contre 90 pour tout tester : 3 scénarios × 30 clients). Les scénarios non conformes ne consomment aucun appel.

```bash
python tests/test_bandit.py                                   # tests sur données synthétiques (sans clé API)
PERSONA_MODE=exhaustive python src/run_eval.py                # référence : chaque scénario × chaque client
python src/bandit_eval.py experiments/outputs/<run_id>.json   # rejeu hors ligne : bandit vs répartition uniforme
```

Le rejeu mesure, sur les réactions enregistrées, combien de fois le bandit retrouve le gagnant du test exhaustif et
combien d'appels il économise, face à une répartition uniforme avec la même règle d'arrêt.
Sur données synthétiques (`tests/test_bandit.py`, *calculé*), les deux méthodes trouvent le gagnant dans 80 à 99 %
des cas avec 14 à 25 appels au lieu de 90 ; le bandit fait aussi bien ou un peu mieux (0 à 4 points). L'essentiel de
l'économie vient de l'arrêt anticipé. Résultat réel : *encore hypothétique* jusqu'au rejeu sur une vraie exécution.
Pour régénérer la base de clients : `python src/make_personas.py --n 30 --seed 2026`.

## Fournisseurs et clés API

Quatre fournisseurs, un compte (et une clé) par fournisseur : les crédits s'achètent par fournisseur, pas par modèle.
Un seul compte par fournisseur suffit pour l'équipe (inviter les membres) ; chaque personne qui lance `run_eval.py`
a besoin des quatre clés dans son `.env`.

| Fournisseur | Modèle (`.env.example`) | Rôles | Clé |
|---|---|---|---|
| Anthropic | `claude-sonnet-5-5` | orchestrateur, générateur « croissance » | `ANTHROPIC_API_KEY` |
| OpenAI | `gpt-4o-mini` | générateur « coûts », personas, baseline | `OPENAI_API_KEY` |
| Google | `gemini-flash-latest` | générateur « marque » | `GEMINI_API_KEY` |
| Mistral | `mistral-large-latest` | auditeur (différent de tous les générateurs) | `MISTRAL_API_KEY` |

Pour revenir à trois fournisseurs, retirer l'entrée `gemini/...` de `GENERATOR_MODELS`.
Les alias `...-latest` évoluent : la version réellement utilisée est notée dans les traces (`model_served`).

## Où sont les preuves

| Livrable demandé | Fichier |
|---|---|
| Cas de test gelés (10) | `tests/cases.json` |
| Résultats baseline vs candidat | `experiments/results.csv` |
| Vérifications automatiques par cas (colonnes `case_pass`, `checks`) | `src/checks.py` |
| Résultats diffusion (seed, steps, temps, mémoire) | `experiments/diffusion_results.csv`, images dans `experiments/diffusion/` |
| Traces (modèle, prompt, version, seed, latence, coût) | `experiments/traces/<run_id>.jsonl` |
| Sorties complètes | `experiments/outputs/<run_id>.json` |
| Rejeu du bandit (gagnant retrouvé, appels économisés) | `experiments/bandit_<run_id>.csv` |
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
| Exécution exhaustive + rejeu du bandit (`bandit_eval.py`) | _à compléter_ |
| Fusion de scénarios | _à compléter_ |
| Interface « Mini Netflix » | _à compléter_ |

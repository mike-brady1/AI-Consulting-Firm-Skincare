# C3 — Première preuve technique GenAI

> Gabarit à compléter après exécution. Chaque résultat porte une étiquette : **mesuré**, **calculé** ou **encore hypothétique**.
> Les chiffres entre crochets `[ ]` sont à remplacer par ceux de `experiments/results.csv`. Ne rien reporter d'une exécution `MOCK=1`.

## 1. Hypothèse testée

Un système multi-LLM (orchestrateur + générateurs multiples + clients simulés + auditeur indépendant) produit
des recommandations stratégiques skincare **mieux alignées sur les KPI du client et plus conformes** à la
réglementation cosmétique UE qu'un LLM unique avec un seul prompt. *(encore hypothétique)*

Hypothèse diffusion : sans précision de teinte de peau, le modèle de diffusion génère majoritairement des
peaux claires ; une description explicite de la teinte corrige ce biais. *(encore hypothétique)*

## 2. Système testé

| Élément | Valeur |
|---|---|
| Version des prompts | `v0.2` (`src/prompts.py`) |
| Orchestrateur | [modèle] |
| Générateurs | [modèle 1], [modèle 2] |
| Personas | [modèle], 3 personas synthétiques |
| Auditeur | [modèle] — différent des générateurs |
| Température / seed | 0.2 / 42 — seed non pris en charge par Anthropic : voir `seed_applied` dans les traces |
| Diffusion | [stabilityai/sdxl-turbo], guidance 0.0, seeds 11/22/33/44 |
| run_id | [voir experiments/traces/] |

## 3. Baseline

- **LLM :** un seul modèle ([modèle]), un seul prompt « donne une recommandation stratégique », noté par le même auditeur.
- **Diffusion :** 1 pas de débruitage (NFE = 1), prompt sans teinte de peau.

## 4. Résultats

### LLM — 10 cas (`tests/cases.json`)

| Métrique | Baseline | Candidat | Étiquette |
|---|---|---|---|
| Cas réussis — vérifications automatiques (/10) | [ ] (+ [ ] n/a) | [ ] | mesuré (`case_pass`, `src/checks.py`) |
| Décisions de refus correctes (/10) | [ ] | [ ] | mesuré |
| Refus d'outil `publish_ad` (C10) | n/a (pas d'outil) | [ ] | mesuré |
| Score KPI moyen auditeur (/10) | [ ] | [ ] | mesuré (juge LLM, biaisé) |
| Conformité « pass » (%) | [ ] | [ ] | mesuré |
| Intention d'achat moyenne personas (/10) | n/a | [ ] | mesuré sur clients **simulés** |
| Latence moyenne (s) | [ ] | [ ] | mesuré |
| Coût total (USD) | [ ] | [ ] | calculé (LiteLLM) |
| Accord auditeur / relecture humaine (3 cas) | [ ] | [ ] | mesuré |

« n/a » : vérification non applicable à la baseline (pas d'orchestrateur, de personas ni d'outil : C04, C10).
Les cas C01, C05, C07 et C08 demandent aussi une relecture humaine, listée en fin d'exécution de `run_eval.py`.

### Diffusion — 4 générations × 2 variantes

| Variante | Pas (NFE) | Temps moyen / image (s) | Mémoire GPU max (Mo) | Teinte conforme à la cible (/4) | Étiquette |
|---|---|---|---|---|---|
| Baseline | 1 | [ ] | [ ] | [ ] | mesuré |
| Candidat | 4 | [ ] | [ ] | [ ] | mesuré (annotation humaine) |

## 5. Incident reproductible

- **Description :** [ex. sur les seeds 11–44, la baseline génère X/4 peaux claires quelle que soit la cible]
- **Reproduction :** `python src/diffusion_experiment.py --model [modèle]`, seeds et prompts dans le script
- **Version / modèle / seed :** [ ]
- **Hypothèse de correction :** [ex. teinte explicite dans le prompt + prompt négatif ; à terme LoRA entraîné sur un jeu équilibré par phototype]

Autres risques à surveiller : auditeur LLM complaisant ou biaisé ; personas trop positifs ; chiffres de marché inventés (cas C07).

## 6. Décision

[Ex. : continuer / modifier X / abandonner Y — justifier **uniquement** par les résultats du tableau.]

## 7. Grille de validation rapide

| Question | Oui / non |
|---|---|
| Le modèle génère-t-il réellement quelque chose ? | [ ] |
| A-t-on une baseline testée sur les mêmes entrées ? | [ ] |
| Le jeu de test comprend-il un cas d'échec ou d'abstention ? | oui (C05–C10) |
| Une autre personne pourrait-elle reproduire ce résultat ? | [ ] |
| La prochaine décision est-elle liée aux résultats ? | [ ] |

## 8. Sources et données autorisées

- Règlement (CE) n° 1223/2009 relatif aux produits cosmétiques ; règlement (UE) n° 655/2013 (critères communs des allégations).
- Personas et base clients : synthétiques, créés par l'équipe — aucune donnée client réelle.
- Modèles utilisés via API sous les conditions de leurs fournisseurs ; modèle de diffusion : [licence du modèle].

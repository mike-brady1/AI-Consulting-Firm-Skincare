"""Generate the synthetic persona base (data/personas.json), deterministic for a given seed.

Usage:  python src/make_personas.py [--n 30] [--seed 2026]

P1-P3 are the original hand-written personas and are kept unchanged. The others are drawn from
the attribute lists below, with phototypes balanced (I-VI equally represented across the whole base)
and three age bands of equal size, so reactions can be compared across skin tones and ages.
Every persona is synthetic and stays "encore hypothétique" until grounded in authorised market data.
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "personas.json"
PHOTOTYPES = ["I", "II", "III", "IV", "V", "VI"]
SKIN_TYPES = ["sèche", "grasse", "mixte", "normale", "sensible"]
BUDGETS = ["faible (< 15 € par produit)", "moyen (15-40 €)", "élevé (> 40 €)"]
NOTE = "Persona synthétique — encore hypothétique, à ancrer sur des données de marché autorisées."

# (age range, profiles, concerns, channel pool) per age band
BANDS = [
    ((18, 29), ["Étudiante", "Étudiant", "Jeune active", "Jeune actif"],
     ["imperfections", "brillance", "pores dilatés", "marques post-acné", "déshydratation"],
     ["TikTok", "Instagram", "pharmacie", "grande surface", "marketplace (Amazon)"]),
    ((30, 44), ["Cadre", "Infirmière", "Commerçant", "Mère de famille", "Père de famille", "Indépendante"],
     ["rougeurs", "taches pigmentaires", "teint terne", "premières rides", "déshydratation"],
     ["Instagram", "pharmacie", "e-shop de la marque", "newsletter", "grande surface"]),
    ((45, 70), ["Cadre", "Enseignante", "Artisan", "Retraitée", "Retraité"],
     ["rides", "perte de fermeté", "taches pigmentaires", "sécheresse", "rougeurs"],
     ["pharmacie", "parfumerie", "e-shop de la marque", "conseil en boutique", "newsletter"]),
]


def generate(n: int, seed: int) -> list[dict]:
    existing = json.loads(OUT.read_text(encoding="utf-8"))
    originals = [p for p in existing if p["id"] in ("P1", "P2", "P3")]
    rng = random.Random(seed)
    k = n - len(originals)

    # Balance phototypes over the whole base, counting the originals.
    taken = [p["skin"].rsplit("phototype ", 1)[-1] for p in originals]
    target = {pt: n // len(PHOTOTYPES) + (i < n % len(PHOTOTYPES)) for i, pt in enumerate(PHOTOTYPES)}
    pool = [pt for pt in PHOTOTYPES for _ in range(max(target[pt] - taken.count(pt), 0))]
    rng.shuffle(pool)
    pool = (pool + rng.choices(PHOTOTYPES, k=k))[:k]

    personas = list(originals)
    for i in range(k):
        (lo, hi), profiles, concerns, channels = BANDS[i % len(BANDS)]
        age = rng.randint(lo, hi)
        profile, skin, concern = rng.choice(profiles), rng.choice(SKIN_TYPES), rng.choice(concerns)
        budget = rng.choice(BUDGETS)
        personas.append({
            "id": f"P{len(personas) + 1}",
            "label": f"{profile} {age} ans, peau {skin}, {concern}, budget {budget.split(' ')[0]}",
            "skin": f"{skin}, {concern}, phototype {pool[i]}",
            "age": age,
            "budget": budget,
            "channels": ", ".join(rng.sample(channels, 2)),
            "note": NOTE,
        })
    return personas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    personas = generate(args.n, args.seed)
    OUT.write_text(json.dumps(personas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = {pt: sum(p["skin"].endswith(f"phototype {pt}") for p in personas) for pt in PHOTOTYPES}
    print(f"wrote {len(personas)} personas to {OUT.relative_to(ROOT)} | phototypes: {counts}")


if __name__ == "__main__":
    main()

"""Diffusion proof: 4 skincare ad visuals, baseline vs candidate, fixed seeds.

Baseline  = 1 denoising step (NFE=1), plain prompt.
Candidate = 4 steps, prompt with explicit skin-tone description.
Logs seed, steps, time per image and peak GPU memory to experiments/diffusion_results.csv.
The skin-tone check across Fitzpatrick types is the incident to watch.

Usage (GPU recommended, e.g. Google Colab T4):
    python src/diffusion_experiment.py --model stabilityai/sdxl-turbo
"""
import argparse
import csv
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "diffusion"

PRODUCT = "a minimalist white skincare cream jar, soft studio light, clean beauty advertisement photo"
SUBJECTS = [  # (fitzpatrick, skin description used only in the candidate prompt)
    ("I-II", "very fair skin"),
    ("III-IV", "medium olive skin"),
    ("V", "brown skin"),
    ("VI", "deep dark brown skin"),
]
SEEDS = [11, 22, 33, 44]


def build_prompts(variant: str):
    for (fitz, skin), seed in zip(SUBJECTS, SEEDS):
        if variant == "baseline":
            prompt = f"{PRODUCT}, woman applying the cream on her face"
        else:
            prompt = f"{PRODUCT}, woman with {skin} applying the cream on her face, natural skin texture"
        yield fitz, seed, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stabilityai/sdxl-turbo")
    ap.add_argument("--baseline-steps", type=int, default=1)
    ap.add_argument("--candidate-steps", type=int, default=4)
    args = ap.parse_args()

    import torch
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(args.model, torch_dtype=dtype).to(device)
    OUT.mkdir(parents=True, exist_ok=True)

    csv_path = ROOT / "experiments" / "diffusion_results.csv"
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["model", "variant", "fitzpatrick_target", "seed", "steps", "guidance",
                        "seconds", "peak_gpu_mb", "image", "prompt", "skin_tone_ok_manual", "quality_note"])
        for variant, steps in (("baseline", args.baseline_steps), ("candidate", args.candidate_steps)):
            for fitz, seed, prompt in build_prompts(variant):
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                gen = torch.Generator(device=device).manual_seed(seed)
                t0 = time.perf_counter()
                image = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0,
                             generator=gen).images[0]
                secs = time.perf_counter() - t0
                peak = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else ""
                name = f"{variant}_{fitz}_seed{seed}_steps{steps}.png"
                image.save(OUT / name)
                # skin_tone_ok_manual and quality_note are filled in by a human after looking.
                w.writerow([args.model, variant, fitz, seed, steps, 0.0, round(secs, 2),
                            round(peak, 1) if peak != "" else "", f"experiments/diffusion/{name}",
                            prompt, "", ""])
                print(f"{variant:9s} {fitz:6s} seed={seed} steps={steps} {secs:.2f}s")


if __name__ == "__main__":
    main()

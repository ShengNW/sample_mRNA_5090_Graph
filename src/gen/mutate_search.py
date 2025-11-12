
import argparse, shutil, subprocess
from pathlib import Path
import numpy as np, pandas as pd, random, yaml
from tqdm import trange
from src.side.dataset import load_manifest

VOCAB = "AUGC"
NUC = list(VOCAB)

def rand_mutate(seq: str, rate: float) -> str:
    s = list(seq); L=len(s); n=max(1, int(L*rate))
    idxs = np.random.choice(L, size=n, replace=False)
    for i in idxs:
        choices = [x for x in NUC if x != s[i]]
        s[i] = random.choice(choices)
    return "".join(s)

def recombine(a: str, b: str) -> str:
    k = np.random.randint(1, len(a))
    return a[:k] + b[k:]

def gc_content(seq: str) -> float:
    s = seq.upper().replace("T","U")
    gc = sum(1 for c in s if c in ("G","C"))
    return gc/max(1,len(s))

def ban_homopolymers(seq: str, max_run: int) -> bool:
    s=seq.upper().replace("T","U"); run=1
    for i in range(1, len(s)):
        if s[i]==s[i-1]:
            run+=1
            if run>max_run: return False
        else:
            run=1
    return True

def has_banned_motif(seq: str, motif: str) -> bool:
    return motif.upper() in seq.upper()

def sample_seed_pairs(n, L5, L3, organ, rng):
    def rnd(L): return "".join(rng.choice(list(NUC), size=L).tolist())
    return [(rnd(L5), rnd(L3), organ) for _ in range(n)]

def score_batch(tmp_csv: str, cfg_path: str, out_csv: str) -> pd.DataFrame:
    import subprocess
    subprocess.run(["python", "-m", "src.side.predict", "--config", cfg_path, "--input", tmp_csv, "--out", out_csv], check=True)
    return pd.read_csv(out_csv)

def maybe_mfe_penalty(seqs):
    exe = shutil.which("RNAfold")
    if exe is None:
        return np.zeros(len(seqs), dtype=np.float32)
    inp = "\\n".join(seqs).encode()
    out = subprocess.run([exe, "--noPS"], input=inp, capture_output=True, check=True).stdout.decode()
    vals = []
    for line in out.splitlines():
        if "(" in line and ")" in line and line.strip().endswith(")"):
            try:
                e = float(line.split("(")[-1].split(")")[0])
            except Exception:
                e = 0.0
            vals.append(e)

    arr = np.array(vals, dtype=np.float32)
    if len(arr)!=len(seqs):
        arr = np.pad(arr, (0,len(seqs)-len(arr)))
    return -arr  # lower MFE => higher reward

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m1_search.yaml")
    ap.add_argument("--predict-config", default="configs/gen_predict.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    pcfg = yaml.safe_load(open(args.predict_config, "r"))
    out_dir = Path(cfg["io"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg["seed"])
    pop_size=int(cfg["population_size"]); steps=int(cfg["steps"])
    mut_rate=float(cfg["mutation_rate"]); recomb_rate=float(cfg["recomb_rate"])
    dataset_dir = pcfg["dataset_dir"]
    man = load_manifest(dataset_dir)
    L5=int(man["shapes"]["utr5"][1]); L3=int(man["shapes"]["utr3"][1])
    target_organ=int(cfg["target_organ"])

    seed_csv = cfg["io"].get("seed_csv")
    if seed_csv and Path(seed_csv).exists():
        seeds = pd.read_csv(seed_csv)
        pool = list(zip(seeds["utr5"], seeds["utr3"], seeds["organ_id"]))
    else:
        pool = sample_seed_pairs(pop_size, L5, L3, target_organ, rng)

    def constraint_ok(u5,u3):
        c = cfg["constraint"]
        gc_ok = c["gc_min"] <= gc_content(u5+u3) <= c["gc_max"]
        hp_ok = ban_homopolymers(u5+u3, c["ban_homopolymers"])
        motif_ok = not any(has_banned_motif(u5+u3, m) for m in c["ban_motifs"])
        return gc_ok and hp_ok and motif_ok

    population = [(u5,u3,org) for (u5,u3,org) in pool if constraint_ok(u5,u3)]
    best = None
    for t in trange(steps, desc="M1-search"):
        children = []
        for _ in range(pop_size):
            if rng.random() < recomb_rate and len(population)>=2:
                a = rng.integers(len(population)); b = rng.integers(len(population))
                u5 = recombine(population[a][0], population[b][0])
                u3 = recombine(population[a][1], population[b][1])
            else:
                p = population[rng.integers(len(population))] if population else sample_seed_pairs(1,L5,L3,target_organ,rng)[0]
                u5 = rand_mutate(p[0], mut_rate); u3 = rand_mutate(p[1], mut_rate)
            if constraint_ok(u5,u3):
                children.append((u5,u3,target_organ))
        cand = population + children
        tmp = out_dir/"_cand.csv"
        pd.DataFrame(cand, columns=["utr5","utr3","organ_id"]).to_csv(tmp, index=False)
        scored = score_batch(str(tmp), "configs/gen_predict.yaml", str(out_dir/"_scored.csv"))
        mfe = maybe_mfe_penalty(scored["utr5"].tolist())
        total = (cfg["score"]["expr_weight"]*scored["pred"].values
                 + cfg["score"]["mfe_weight"]*mfe)
        scored["total"] = total
        scored = scored.sort_values("total", ascending=False).head(pop_size)
        population = list(zip(scored["utr5"], scored["utr3"], scored["organ_id"]))
        if best is None or scored.iloc[0]["total"] > best["total"]:
            best = scored.iloc[0].to_dict()
    topk = int(cfg["io"]["topk_export"])
    out_csv = out_dir/"m1_topk.csv"
    pd.DataFrame(population, columns=["utr5","utr3","organ_id"]).head(topk).to_csv(out_csv, index=False)
    print(f"[M1] Wrote {out_csv}")

if __name__ == "__main__":
    main()

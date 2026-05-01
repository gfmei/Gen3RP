# Gen3RP


## Run
From the Gen3R repo root:

```bash
mkdir -p tools logs
sbatch run_gen3r_re10k_exact_eval.sh
```


For a quick debug on one sample:
```
python tools/prepare_gen3r_re10k_exact.py \
  --root /leonardo_scratch/fast/EUHPC_D30_012/re10k_preprocessed_subsampled_test \
  --eval_json re10k_eval_v2.json \
  --out_root /leonardo_work/EUHPC_D30_012/results/gen3r_re10k_variant_exact/prepared \
  --difficulty easy \
  --num_context_views 2 \
  --context_selection even \
  --max_samples 1 \
  --prompt "a realistic indoor room"
```

Check the camera JSON:

```angular2html
python - <<'PY'
import json
from pathlib import Path

root = Path("/leonardo_work/EUHPC_D30_012/results/gen3r_re10k_variant_exact/prepared")
p = next(root.rglob("cameras_exact_targets.json"))

with p.open() as f:
    d = json.load(f)

print(p)
print(d.keys())
print("num extrinsics:", len(d["extrinsics"]))
print("num intrinsics:", len(d["intrinsics"]))
print("extrinsic shape:", len(d["extrinsics"][0]), len(d["extrinsics"][0][0]))
print("intrinsic shape:", len(d["intrinsics"][0]), len(d["intrinsics"][0][0]))
print("target ids:", d["target_view_ids"])
PY
```

Expected:
```angular2html
dict_keys(['extrinsics', 'intrinsics', ...])
num extrinsics: 2
num intrinsics: 2
extrinsic shape: 4 4
intrinsic shape: 3 3
target ids: [43, 44]
```
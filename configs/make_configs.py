import json
import os

with open("config.json", "r") as f:
  cfg = json.load(f)

data_dir, _ = os.path.split(cfg['data_path'])
save_dir, _ = os.path.split(cfg['summary_path'])

for dataset in os.listdir("../data/pickles"):
  cfg_iter = cfg
  name, _ = os.path.splitext(dataset)
  if "50_salads" in dataset:
    cfg_iter['data_path'] = os.path.join(data_dir, dataset)
    cfg_iter['summary_path'] = f"../runs/{name}"
    with open(f"cfg_{name}.json", "w") as f:
      json.dump(cfg_iter, f)
    cfg_iter['enable_matrix'] = False
    with open(f"cfg_{name}_no_mat.json", "w") as f:
      json.dump(cfg_iter, f)
    
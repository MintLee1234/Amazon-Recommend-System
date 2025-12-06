## Directory Overview
- `src/`: Training scripts, model definitions (`models/PGL.py`), configuration files, and utility functions.
- `raw_data/`: Contain raw data download form Amazon Review dataset website
- `data/`: Contains prepared multimodal features and interaction data, organized by dataset (`baby/`).
- `preprocessing/`: Scripts and notebooks for constructing features from raw Amazon/ datasets.
- `saved_model/`: Save the weight of the best model.

## Environment Setup
OS: Linux (Recommend)

### Pip / Virtualenv (Lightweight)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- Additional GPU/graph learning packages (PyTorch, torch-geometric, DGL) may be required.
- Ensure the shell's `PYTHONPATH` contains `src/`:
  ```bash
  export PYTHONPATH=$(pwd)/src:$PYTHONPATH
  ```

## Data Preparation
See /preprocessing/README.md

### Download Prepared Datasets
1. Download the processed datasets.
2. Extract them into `data/`. Example (`baby` dataset):
   ```
   Amazon-Recommend-System/
   ├── data/
   │   └── baby/
   │       ├── baby.inter           # Interaction file: userID, itemID, x_label...
   │       ├── image_feat.npy       # Pre‑extracted image features
   │       ├── text_feat.npy        # Pre‑extracted text features
   │       ├── user_graph_dict.npy  # Optional
   │       ├── item_graph_dict.npy  # Optional
   │       └── ...
   ```
3. Corresponding settings appear in `src/configs/dataset/baby.yaml`.
4. `x_label` controls train/valid/test splits (0/1/2). Filtering cold‑start users can be configured in `overall.yaml`.

### Using Custom Data
1. Follow `preprocessing/README.md` to clean data, split sets, extract features, and remap IDs.
2. Place `.inter`, `*feat.npy`, and graph files into `data/<your_dataset>/`.
3. Add a new YAML config under `src/configs/dataset/`.
4. Run the model using:
   ```bash
   -d <your_dataset>
   ```

## Running PGL
1. Move to the `src/` directory and activate your environment:
   ```bash
   cd src
   export PYTHONPATH=$(pwd):$PYTHONPATH
   ```
2. Example: train on GPU 0:
   ```bash
   python main.py -m PGL -d baby -g 0
   ```
3. Logs are stored in `src/log/`, checkpoints in `saved_model/`.

## Hyperparameter Search
Hyperparameters are defined in `src/configs/model/PGL.yaml` and managed via `utils/quick_start.py`.

1. Edit search parameters:
   ```yaml
   learning_rate: [0.0005, 0.001]
   diff_weight: [0.25, 0.5]
   timesteps: [10, 20]
   ```
2. Add them to:
   ```yaml
   hyper_parameters: ["diff_weight", "ssl_weight", "w1", "w2", "walk_length", "timesteps", "learning_rate", "choosing_tmp"]
   ```
3. Run training; all combinations will be evaluated.
4. Logs include full results and the best parameter set.

## Troubleshooting
- **Data path issues** → Always run from `src/` or specify `--data_path`.
- **Graph cache mismatch** → Delete cached adjacency files after changing `image_knn_k` or `text_knn_k`.
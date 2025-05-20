# NAAS: Post-hoc Alignment of Knowledge Graph Embeddings

**NAAS** (Neural Adaptive Alignment Space) is a post-hoc method for aligning entity and relation embeddings from independently trained Knowledge Graph Embedding (KGE) models. NAAS enables the use of embeddings for downstream tasks—such as entity alignment and link prediction—**without retraining or joint embedding**.

---

##  Motivation

Traditional joint-training methods like MTransE, BootEA, or KDCoE require modifying the KGE training process. NAAS takes a different approach: it aligns embeddings **after** training, using neural adaptation to preserve structural semantics across knowledge graphs (KGs).

---

##  Code and Data

- **Embeddings**: Trained using the [DICE Embedding Framework](https://github.com/dice-group/dice-embeddings)
- **Datasets**:
  - [OpenEA benchmark](https://www.dropbox.com/scl/fi/lo69wjm1f37qiik59kmg8/OpenEA_dataset_v1.1.zip)
  - [Zenodo – DBpedia-Wikidata](https://zenodo.org/records/7566020)
- **Repo**: [Anonymous GitHub](https://github.com/anon-kgalignment/embedding_alignment) (to be made public after review)

---

##  Files Required

Each embedding folder must contain:

- `model.pt` – Trained model
- `entity_to_idx.p` – Entity-to-ID mapping (Pickle)
- `relation_to_idx.p` – Relation-to-ID mapping (Pickle)
- `configuration.json` – Model configuration

---

##  Installation

1. Train your embeddings using [DICE](https://github.com/dice-group/dice-embeddings)
2. Download or create alignment dictionaries
3. Use this repo for post-hoc alignment via Procrustes or NAAS

---

##  Pipeline Overview

1. **Procrustes Alignment (`run_procrustes.py`)**  
   Aligns independently trained embeddings using Orthogonal Procrustes.

   - Output:
     - A single folder specified via `--output_folder`, e.g., `procrustes_results/`
     - Inside this folder:
       - `new_model/` – Procrustes-aligned embeddings
       - `previous_model/` – Initially identical copy for comparison purposes

    **This same output folder is used as `--output_dir` in both NAAS scripts**  
   NAAS uses these two subfolders to perform Bayesian optimization, evaluate alignment models, and compare baseline vs improved results.

2. **Entity Alignment (`NAAS_entity_alignment.py`)**  
   Uses the Procrustes output folder to train and evaluate NAAS with both `new_model/` and `previous_model/`.

3. **Link Prediction (`NAAS_link_prediction.py`)**  
   Also uses the same output folder to evaluate alignment quality through link prediction metrics.

---

##  Scripts

- `run_procrustes.py` – Runs Procrustes alignment
- `NAAS_entity_alignment.py` – Neural model for entity alignment
- `NAAS_link_prediction.py` – Neural model for link prediction

---

##  Example Usage
###  Step 1: Procrustes Alignment
```bash
python run_procrustes.py \
  --embedding_folder /path/to/embedding_folder \
  --alignment_dict_path /path/to/pre_aligned_alignment_dict \
  --alignmentlimes_dict_path /path/to/limes_alignment_dict \
  --output_folder /path/to/output_folder 
  ```

###  Step 2: NAAS Entity Alignment
```bash
  python NAAS_entity_alignment.py \
  --source_dir /path/to/source_embeddings \
  --target_dir /path/to/target_embeddings \
  --train_alignment_folder /path/to/train_alignment_links \
  --val_alignment_folder /path/to/val_alignment_links \
  --test_alignment_folder /path/to/test_alignment_links \
  --output_dir /path/to/save_results
  ```

###  Step 3: NAAS Link Prediction
```bash
python NAAS_link_prediction.py \
  --source_dir /path/to/source_embeddings \
  --target_dir /path/to/target_embeddings \
  --alignment_dir /path/to/procrustes_output \
  --test_triples /path/to/procrustes_output/new_model/test_triples.txt \
  --output_dir /path/to/save_results
  ```

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os
import gc
import shutil
import pickle
import logging
import random
import time
import argparse
import glob
import sys
import heapq

from rdflib import Graph
from tqdm import tqdm
from itertools import product
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset
from skopt import gp_minimize
from skopt.space import Real
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from multiprocessing import Pool
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

best_mrr = -1
best_model_state_dict = None
best_model_params = None
best_metrics = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entity Alignment Training Script")
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--train_alignment_folder', type=str, required=True)
    parser.add_argument('--val_alignment_folder', type=str, required=True)
    parser.add_argument('--test_alignment_folder', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    directory_1 = args.source_dir
    directory_2 = args.target_dir
    train_alignment_path = args.train_alignment_folder
    val_alignment_path = args.val_alignment_folder
    test_alignment_path = args.test_alignment_folder
    output_dir = args.output_dir

    def find_dataset_folders(embedding_folder):
        """Find and return paths for dataset folders in the embedding folder."""
        all_items = sorted(os.listdir(embedding_folder))

        # Filter for directories that represent datasets
        dataset_folders = [item for item in all_items if os.path.isdir(
            os.path.join(embedding_folder, item))]

        # Check if there are exactly 2 datasets
        if len(dataset_folders) != 2:
            raise ValueError(
                f"Expected exactly 2 datasets in {embedding_folder}, found {len(dataset_folders)}.")

        # Return the paths of the dataset folders
        dataset1_path = os.path.join(embedding_folder, dataset_folders[0])
        dataset2_path = os.path.join(embedding_folder, dataset_folders[1])

        return dataset1_path, dataset2_path

    def extract_files_from_directory(directory):
        """Retrieve required files (model, entity_to_idx, relation_to_idx) from a directory."""
        model_path = os.path.join(directory, "model.pt")

        # Possible file formats for entity_to_idx and relation_to_idx
        entity_to_id_options = [
            os.path.join(directory, "entity_to_idx.p"),
            os.path.join(directory, "entity_to_idx"),
            os.path.join(directory, "entity_to_idx.csv")
        ]

        relation_to_id_options = [
            os.path.join(directory, "relation_to_idx.p"),
            os.path.join(directory, "relation_to_idx"),
            os.path.join(directory, "relation_to_idx.csv")
        ]

        # Check for existing entity_to_idx file
        entity_to_id_path = next(
            (path for path in entity_to_id_options if os.path.exists(path)), None)
        if entity_to_id_path is None:
            logging.error(
                f"Entity-to-ID file not found in {directory}. Checked: {entity_to_id_options}")
            raise FileNotFoundError(
                f"Missing entity_to_idx file in {directory}.")

        # Check for existing relation_to_idx file
        relation_to_id_path = next(
            (path for path in relation_to_id_options if os.path.exists(path)), None)
        if relation_to_id_path is None:
            logging.error(
                f"Relation-to-ID file not found in {directory}. Checked: {relation_to_id_options}")
            raise FileNotFoundError(
                f"Missing relation_to_idx file in {directory}.")

        logging.debug(
            f"Files found: Model={model_path}, Entity-to-ID={entity_to_id_path}, Relation-to-ID={relation_to_id_path}")

        return model_path, entity_to_id_path, relation_to_id_path

    def load_embeddings(model_path, entity_to_id_path, relation_to_id_path):
        """Load embeddings and mappings for entities and relations."""

        logging.info(f"Loading model weights from: {model_path}")
        model_weights = torch.load(
            model_path, map_location='cpu', weights_only=True)

        # Load entity and relation embeddings
        entity_embeddings = model_weights['entity_embeddings.weight'].cpu(
        ).detach().numpy()
        relation_embeddings = model_weights['relation_embeddings.weight'].cpu(
        ).detach().numpy()

        logging.info(f"Entity embeddings shape: {entity_embeddings.shape}")
        logging.info(f"Relation embeddings shape: {relation_embeddings.shape}")

        # Handle entity-to-ID mapping
        try:
            entity_df = pd.read_csv(entity_to_id_path)
            entity_to_id = dict(zip(entity_df.index, entity_df["entity"]))
            logging.info(
                f"Successfully loaded entity mappings from Parquet (without extension): {entity_to_id_path}")

        except Exception as e:
            raise ValueError(
                f"Could not load entity-to-ID mapping from {entity_to_id_path}: {str(e)}")

        # Handle relation-to-ID mapping similarly
        try:
            relation_df = pd.read_csv(relation_to_id_path)
            relation_to_id = dict(
                zip(relation_df.index, relation_df["relation"]))
            logging.info(
                f"Successfully loaded relation mappings from Parquet (without extension): {relation_to_id_path}")

        except Exception as e:
            raise ValueError(
                f"Could not load relation-to-ID mapping from {relation_to_id_path}: {str(e)}")

        # Match embeddings to URIs using entity index
        sorted_entities = [entity_to_id[i]
                           for i in range(len(entity_embeddings))]
        sorted_relations = [relation_to_id[i]
                            for i in range(len(relation_embeddings))]

        # Ensure lengths match
        if len(entity_embeddings) != len(sorted_entities):
            logging.warning(
                f"Mismatch: {len(entity_embeddings)} entity embeddings vs. {len(sorted_entities)} entities. Fixing it...")
            sorted_entities = sorted_entities[:len(entity_embeddings)]

        if len(relation_embeddings) != len(sorted_relations):
            logging.warning(
                f"Mismatch: {len(relation_embeddings)} relation embeddings vs. {len(sorted_relations)} relations. Fixing it...")
            sorted_relations = sorted_relations[:len(relation_embeddings)]

        # Create DataFrames
        entity_embeddings_df = pd.DataFrame(
            entity_embeddings, index=sorted_entities)
        relation_embeddings_df = pd.DataFrame(
            relation_embeddings, index=sorted_relations)

        return entity_embeddings_df, relation_embeddings_df

    directory_1 = args.source_dir
    directory_2 = args.target_dir

    # Extract files for both directories
    model_path_1, entity_to_id_1, relation_to_id_1 = extract_files_from_directory(
        directory_1)
    model_path_2, entity_to_id_2, relation_to_id_2 = extract_files_from_directory(
        directory_2)
    entity_embeddings_df1, relation_embeddings_df1 = load_embeddings(
        model_path_1, entity_to_id_1, relation_to_id_1)
    entity_embeddings_df2, relation_embeddings_df2 = load_embeddings(
        model_path_2, entity_to_id_2, relation_to_id_2)

    def clean_uri(uri):
        """ Remove angle brackets, extra >>, and << from URIs. """
        return uri.replace("<<", "").replace(">>", "").replace("<", "").replace(">", "").strip()

    def build_alignment_dict(folder_path):
        """
        Build an alignment dictionary from all files in a folder using only subject and object URIs.

        Args:
            folder_path (str): Path to the folder containing alignment files (.nt, .ttl, .txt).

        Returns:
            dict: A dictionary mapping subject URIs to object URIs.
        """
        alignment_dict = {}

        if not os.path.exists(folder_path):
            logging.error(f"Folder '{folder_path}' does not exist.")
            return alignment_dict

        if not os.listdir(folder_path):
            logging.warning(
                f"Alignment folder '{folder_path}' is empty. Skipping processing.")
            return alignment_dict

        file_paths = [os.path.join(folder_path, file)
                      for file in os.listdir(folder_path)]

        for file_path in file_paths:
            extension = os.path.splitext(file_path)[1].lower()

            try:
                if extension in ['.nt', '.ttl']:
                    # Parse RDF-based files
                    g = Graph()
                    g.parse(file_path, format='nt' if extension ==
                            '.nt' else 'ttl')

                    for subj, pred, obj in g:
                        subj = clean_uri(str(subj))
                        pred = clean_uri(str(pred))
                        obj = clean_uri(str(obj))

                        if "sameAs" in pred:
                            alignment_dict[subj] = obj
                        else:
                            logging.warning(
                                f"Skipping triple with unexpected predicate: {subj} {pred} {obj}")

                elif extension in ['', '.txt']:
                    # Parse text-based alignment files
                    with open(file_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 4 and "sameAs" in parts[1]:
                                entity1 = clean_uri(parts[0])
                                entity2 = clean_uri(parts[2])
                                alignment_dict[entity1] = entity2
                            elif len(parts) == 2:
                                entity1 = clean_uri(parts[1])
                                entity2 = clean_uri(parts[0])
                                alignment_dict[entity1] = entity2
                            else:
                                logging.warning(
                                    f"Skipping line (unexpected format): {line.strip()}")

                else:
                    logging.warning(f"Unsupported file type: {file_path}")

            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")

        return alignment_dict

    def remove_brackets_from_indices(embeddings_df):
        """Remove < and > from each index in the DataFrame."""
        cleaned_index = [uri.strip('<>') for uri in embeddings_df.index]
        embeddings_df.index = cleaned_index
        return embeddings_df

    entity_embeddings1 = remove_brackets_from_indices(entity_embeddings_df1)
    entity_embeddings2 = remove_brackets_from_indices(entity_embeddings_df2)

    relation_embeddings1 = remove_brackets_from_indices(
        relation_embeddings_df1)
    relation_embeddings2 = remove_brackets_from_indices(
        relation_embeddings_df2)

    def clean_dict(input_dict):
        """
        Cleans a dictionary by removing angle brackets and any trailing characters like '> .'
        from both keys and values.
        """
        return {
            k.strip('<>'): v.strip('<>') if isinstance(v, str) else v
            for k, v in input_dict.items()
        }

    train_dict = build_alignment_dict(args.train_alignment_folder)
    val_dict = build_alignment_dict(args.val_alignment_folder)
    test_dict = build_alignment_dict(args.test_alignment_folder)

    cleaned_alignment_dict_train = clean_dict(train_dict)
    cleaned_alignment_dict_val = clean_dict(val_dict)
    cleaned_alignment_dict_test = clean_dict(test_dict)

    def create_train_val_test_matrices_from_links(train_links, val_links, test_links, entity_embeddings1, entity_embeddings2):
        """Generates train, validation, and test matrices using pre-defined alignment links."""

        def filter_links(links, name):
            filtered = []
            skipped = 0

            for link in links:
                if len(link) != 2:
                    skipped += 1
                    continue
                e1, e2 = link
                if e1 in entity_embeddings1.index and e2 in entity_embeddings2.index:
                    filtered.append((e1, e2))
            if skipped > 0:
                logging.warning(
                    f"{skipped} malformed links were skipped in {name}_links.")
            logging.info(
                f"{len(filtered)} valid links retained in {name}_links.")
            return filtered

        filtered_train_links = filter_links(train_links, "train")
        filtered_val_links = filter_links(val_links, "val")
        filtered_test_links = filter_links(test_links, "test")

        # Extract train embeddings
        S_train = entity_embeddings1.loc[[
            e1 for e1, _ in filtered_train_links]].values if filtered_train_links else None
        T_train = entity_embeddings2.loc[[
            e2 for _, e2 in filtered_train_links]].values if filtered_train_links else None

        # Extract val embeddings
        S_val = entity_embeddings1.loc[[
            e1 for e1, _ in filtered_val_links]].values if filtered_val_links else None
        T_val = entity_embeddings2.loc[[
            e2 for _, e2 in filtered_val_links]].values if filtered_val_links else None

        # Extract test embeddings
        S_test = entity_embeddings1.loc[[
            e1 for e1, _ in filtered_test_links]].values if filtered_test_links else None
        T_test = entity_embeddings2.loc[[
            e2 for _, e2 in filtered_test_links]].values if filtered_test_links else None

        return S_train, T_train, S_val, T_val, S_test, T_test

    train_links = [(en, de) for de, en in cleaned_alignment_dict_train.items()]
    val_links = [(en, de) for de, en in cleaned_alignment_dict_val.items()]
    test_links = [(en, de) for de, en in cleaned_alignment_dict_test.items()]

    S_train, T_train, S_val, T_val, S_test, T_test = create_train_val_test_matrices_from_links(
        train_links, val_links, test_links, entity_embeddings1, entity_embeddings2)

    def normalize_and_scale(data, reference_data=None):
        """
        Normalize and scale data using its mean and standard deviation, or those of a reference dataset.

        Parameters:
        - data: ndarray, the data to be normalized.
        - reference_data: ndarray, the reference data used for calculating mean and scale. If None, uses data itself.

        Returns:
        - normalized_data: ndarray, the normalized data.
        - mean: ndarray, the mean used for normalization.
        - scale: float, the scale used for normalization.
        """
        if reference_data is None:
            reference_data = data

        mean = reference_data.mean(axis=0)
        scale = np.sqrt(((reference_data - mean) ** 2).sum() /
                        reference_data.shape[0])
        normalized_data = (data - mean) / scale

        return normalized_data, mean, scale


    S_train_normalized, S_train_mean, S_train_scale = normalize_and_scale(
        S_train)
    T_train_normalized, T_train_mean, T_train_scale = normalize_and_scale(
        T_train)

    S_test = pd.DataFrame(np.array(S_test))
    T_test = pd.DataFrame(np.array(T_test))
    S_val = np.array(S_val)
    T_val = np.array(T_val)
    S_test_normalized, _, _ = normalize_and_scale(
        S_test, reference_data=S_train)
    T_test_normalized, _, _ = normalize_and_scale(
        T_test, reference_data=T_train)
    S_val_normalized, _, _ = normalize_and_scale(S_val, reference_data=S_train)
    T_val_normalized, _, _ = normalize_and_scale(T_val, reference_data=T_train)
    S_test_normalized = S_test_normalized.to_numpy() if hasattr(
        S_test_normalized, 'to_numpy') else S_test_normalized
    T_test_normalized = T_test_normalized.to_numpy() if hasattr(
        T_test_normalized, 'to_numpy') else T_test_normalized

    class SharedSpaceAlignmentNN(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(SharedSpaceAlignmentNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.activation = nn.GELU()
            self.fc2 = nn.Linear(hidden_dim, input_dim)
            self.shared_layer = nn.Linear(input_dim, input_dim)
            self.alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True))

        def forward(self, entities):
            transformed = self.fc2(self.activation(self.fc1(entities)))
            shared_space = self.shared_layer(transformed)
            alpha_scaled = torch.sigmoid(self.alpha)
            aligned_embeddings = (1 - alpha_scaled) * \
                entities + alpha_scaled * shared_space
            return aligned_embeddings, shared_space

    def loss_fn(S_aligned, T_aligned, S_shared, T_shared, S_train_tensor, T_train_tensor, w1, w2, w3, w4, return_details=False):
        mse_loss = nn.MSELoss()

        def cosine_similarity_loss(original, transformed):
            cos_sim = F.cosine_similarity(original, transformed, dim=1)
            return mse_loss(cos_sim, torch.ones_like(cos_sim))

        def contrastive_loss(S_shared, T_shared, margin=1.0, k=20, batch_size=512):
            batch_size_total = S_shared.size(0)
            all_pos = []
            all_mean_neg = []

            for start in range(0, batch_size_total, batch_size):
                end = min(start + batch_size, batch_size_total)
                S_batch = S_shared[start:end]  # shape: [B, D]
                sim_matrix = F.cosine_similarity(S_batch.unsqueeze(
                    1), T_shared.unsqueeze(0), dim=2)  # shape: [B, N]

                pos = torch.diagonal(sim_matrix[:, start:end]) if end - \
                    start == batch_size else sim_matrix[:, start:end].diagonal()
                neg = sim_matrix.clone()
                neg[:, start:end] = -1  # remove positives

                topk_neg, _ = torch.topk(neg, k=min(
                    k, batch_size_total - 1), dim=1)
                mean_neg = topk_neg.mean(dim=1)

                all_pos.append(pos)
                all_mean_neg.append(mean_neg)

            pos_all = torch.cat(all_pos)
            mean_neg_all = torch.cat(all_mean_neg)
            const_loss = F.relu(margin - pos_all + mean_neg_all).mean()

            return const_loss, pos_all.mean().item(), mean_neg_all.mean().item()

        structure_loss = mse_loss(
            S_aligned, S_train_tensor) + mse_loss(T_aligned, T_train_tensor)
        contrastive_loss_val, pos_mean, mean_hard_neg = contrastive_loss(
            S_shared, T_shared)
        directional_loss = cosine_similarity_loss(
            S_train_tensor, S_aligned) + cosine_similarity_loss(T_train_tensor, T_aligned)
        magnitude_loss = mse_loss(torch.norm(S_train_tensor, dim=1), torch.norm(S_aligned, dim=1)) + \
            mse_loss(torch.norm(T_train_tensor, dim=1),
                     torch.norm(T_aligned, dim=1))

        weight_sum = w1 + w2 + w3 + w4
        if weight_sum == 0:
            total_loss = structure_loss + contrastive_loss_val + \
                directional_loss + magnitude_loss
        else:
            total_loss = (w1 * structure_loss + w2 * contrastive_loss_val +
                          w3 * directional_loss + w4 * magnitude_loss)

        if return_details:
            return total_loss, contrastive_loss_val.item(), pos_mean, mean_hard_neg
        else:
            return total_loss

    # Function to Train Alignment Model

    def train_alignment_model(S_train, T_train, S_val, T_val, input_dim, hidden_dim, epochs, lr,
                              w1, w2, w3, w4, log_path, swa_start=30):
        # Initialize base model and optimizer
        base_model = SharedSpaceAlignmentNN(input_dim, hidden_dim)
        base_optimizer = torch.optim.Adam(base_model.parameters(), lr=lr)

        # Create the SWA model and scheduler
        swa_model = AveragedModel(base_model)
        swa_scheduler = SWALR(base_optimizer, swa_lr=lr)

        S_train_tensor = torch.tensor(S_train, dtype=torch.float32)
        T_train_tensor = torch.tensor(T_train, dtype=torch.float32)
        S_val_tensor = torch.tensor(S_val, dtype=torch.float32)
        T_val_tensor = torch.tensor(T_val, dtype=torch.float32)

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, "w")

        for epoch in range(epochs):
            base_model.train()
            base_optimizer.zero_grad()

            S_aligned, S_shared = base_model(S_train_tensor)
            T_aligned, T_shared = base_model(T_train_tensor)

            total_loss_val, contrastive_loss_val, pos_mean, mean_hard_neg = loss_fn(
                S_aligned, T_aligned, S_shared, T_shared,
                S_train_tensor, T_train_tensor,
                w1, w2, w3, w4,
                return_details=True
            )

            total_loss_val.backward()
            base_optimizer.step()

            # Validation every epoch (metrics always saved)
            base_model.eval()
            with torch.no_grad():
                S_val_aligned, _ = base_model(
                    S_val_tensor)  # S_val_neighbors_tensor
                T_val_aligned, _ = base_model(
                    T_val_tensor)  # S_val_neighbors_tensor

                _, hits_at_k_val, mean_rank_val, mrr_val = greedy_alignment(
                    embeds1=S_val_aligned.numpy(),
                    embeds2=T_val_aligned.numpy(),
                    top_k=[1, 5, 10, 50],
                    threads_num=1,
                    metric='inner',
                    normalize=False
                )

            model_path = os.path.join(os.path.dirname(
                log_path), f"alignment_model_epoch{epoch+1}.pt")
            torch.save(swa_model.state_dict(), model_path)

            # Save metrics every epoch
            metrics = {
                "epoch": epoch + 1,
                "training_loss": total_loss_val.item(),
                "val_Hits@1": hits_at_k_val[0],
                "val_Hits@5": hits_at_k_val[1],
                "val_Hits@10": hits_at_k_val[2],
                "val_Hits@50": hits_at_k_val[3],
                "val_MRR": mrr_val,
                "pos_mean": pos_mean,
                "mean_hard_neg": mean_hard_neg,
                "contrastive_loss": contrastive_loss_val
            }

            log_file.write(json.dumps(metrics) + "\n")
            log_file.flush()

            if epoch >= swa_start:
                swa_model.update_parameters(base_model)
                swa_scheduler.step()

            if epoch % 1 == 0:
                print(f"[Epoch {epoch}] Total Loss: {total_loss_val.item():.4f}, Contrastive Loss: {contrastive_loss_val:.4f}, Pos Mean: {pos_mean:.4f}, Mean Hard Neg: {mean_hard_neg:.4f}")

        log_file.close()

        # === Finalize SWA model and save it ===
        update_bn(DataLoader(TensorDataset(
            S_train_tensor), batch_size=64), swa_model)

        final_swa_model_path = os.path.join(
            os.path.dirname(log_path), "final_swa_model.pt")
        torch.save(swa_model.state_dict(), final_swa_model_path)
        print(f" Final SWA model saved to {final_swa_model_path}")

        return swa_model, total_loss_val.item(), mrr_val

    def load_json(p: str) -> dict:
        with open(p, 'r') as r:
            args = json.load(r)
        return args

    input_dim = S_train_normalized.shape[1]

    def greedy_alignment(embeds1, embeds2, top_k, threads_num, metric='inner', normalize=False, csls_k=0, accurate=False):
        entity_num = embeds1.shape[0]
        step = (entity_num + threads_num - 1) // threads_num
        ranges = [(i, min(i + step, entity_num))
                  for i in range(0, entity_num, step)]

        # Add embeds1, embeds2, top_k, etc., to each argument tuple
        args = [(start, end, embeds1, embeds2, top_k, metric, normalize)
                for start, end in ranges]

        with Pool(threads_num) as p:
            parts = p.starmap(_one_thread, args)

        alignment_result = [x for part in parts for x in part]

        hits = []
        for k in top_k:
            hit_k = np.mean([1 if i in alignment_result[i][:k]
                            else 0 for i in range(len(alignment_result))])
            hits.append(hit_k)

        ranks = [alignment_result[i].index(i) + 1 if i in alignment_result[i] else len(
            alignment_result[i]) for i in range(len(alignment_result))]
        mr = np.mean(ranks)
        mrr = np.mean([1.0 / r for r in ranks])

        return alignment_result, hits, mr, mrr

    def test(embeds1, embeds2, mapping, top_k, threads_num, metric='inner', normalize=False, csls_k=0, accurate=True):
        if mapping is None:
            alignment_rest_12, hits1_12, mr_12, mrr_12 = greedy_alignment(embeds1, embeds2, top_k, threads_num,
                                                                          metric, normalize, csls_k, accurate)
        else:
            test_embeds1_mapped = np.matmul(embeds1, mapping)
            alignment_rest_12, hits1_12, mr_12, mrr_12 = greedy_alignment(test_embeds1_mapped, embeds2, top_k, threads_num,
                                                                          metric, normalize, csls_k, accurate)
        return alignment_rest_12, hits1_12, mr_12, mrr_12

    print("Starting Bayesian optimization...")

    ###### BAYESIAN#####
    bayesian_start_time = time.time()

    # We only optimize w1, w2, w3, and enforce w4 = 1 - (w1 + w2 + w3)
    search_space = [Real(0, 1), Real(0, 1), Real(0, 1)]  # Only 3 variables

    def objective(params):
        global best_mrr, best_model_state_dict, best_model_params, best_metrics

        trial_time = int(time.time())
        trial_dir = os.path.join(args.output_dir, f"trial_{trial_time}")

        os.makedirs(trial_dir, exist_ok=True)

        log_path = os.path.join(trial_dir, "training_log.jsonl")

        # Convert params to soft weights
        logits = torch.tensor(params, dtype=torch.float32)
        soft_weights = torch.nn.functional.softmax(logits, dim=0).numpy()
        w1, w2, w3 = soft_weights
        w4 = max(1.0 - (w1 + w2 + w3), 0.05)
        scale = 1.0 / (w1 + w2 + w3 + w4)
        w1 *= scale
        w2 *= scale
        w3 *= scale
        w4 *= scale

        print(
            f"\n[Bayesian Iteration] weights: w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}, w4={w4:.4f}")

        log_path = os.path.join(trial_dir, "training_log.jsonl")

        alignment_model, final_loss, mrr_val = train_alignment_model(
            S_train_normalized, T_train_normalized, S_val_normalized, T_val_normalized,
            input_dim, 256, 10, 0.003,  # hidden_dim, epochs, lr
            w1, w2, w3, w4,
            log_path
        )

        # Save best metrics
        if mrr_val > best_mrr:
            best_mrr = mrr_val
            best_model_state_dict = alignment_model.state_dict()
            best_model_params = (w1, w2, w3, w4)
            best_metrics = {
                "final_loss": final_loss,
                "weights": {"w1": w1, "w2": w2, "w3": w3, "w4": w4}
            }

        return -mrr_val  # For maximization

    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        acq_func="EI",  # Expected improvement
        n_calls=5,
        n_random_starts=5,
        random_state=42
    )

    # === After Bayesian optimization, find and load best SWA model ===
    root_trials_dir = args.output_dir
    trial_folders = sorted(glob.glob(os.path.join(root_trials_dir, "trial_*")))

    best_mrr = -1
    best_trial_dir = None
    best_swa_model_path = None

    for trial_dir in trial_folders:
        log_path = os.path.join(trial_dir, "training_log.jsonl")
        swa_model_path = os.path.join(trial_dir, "final_swa_model.pt")

        if not os.path.exists(log_path) or not os.path.exists(swa_model_path):
            continue

        with open(log_path, "r") as f:
            lines = f.readlines()
            if not lines:
                continue
            try:
                final_log = json.loads(lines[-1])
                mrr = final_log.get("val_MRR", -1)

                if mrr > best_mrr:
                    best_mrr = mrr
                    best_trial_dir = trial_dir
                    best_swa_model_path = swa_model_path

            except Exception as e:
                print(f"⚠️ Failed to parse log in {log_path}: {e}")
                continue

    if best_trial_dir is None:
        raise ValueError(" No valid SWA model logs found.")

    print(f" Best SWA model MRR = {best_mrr:.4f} in: {best_trial_dir}")

    # Load the best SWA model
    model = SharedSpaceAlignmentNN(input_dim, 256)
    swa_model = AveragedModel(model)
    swa_model.load_state_dict(torch.load(
        best_swa_model_path, map_location="cpu"))
    swa_model.eval()

    print(f"Loaded best SWA model from {best_swa_model_path}")

    # Align test embeddings
    model.eval()
    with torch.no_grad():
        S_test_tensor = torch.tensor(S_test_normalized, dtype=torch.float32)
        T_test_tensor = torch.tensor(T_test_normalized, dtype=torch.float32)
        # S_test_neighbors_tensor = torch.tensor(S_test_neighbors, dtype=torch.float32)
        # T_test_neighbors_tensor = torch.tensor(T_test_neighbors, dtype=torch.float32)
        S_aligned, _ = swa_model(S_test_tensor)
        T_aligned, _ = swa_model(T_test_tensor)

    # Evaluate
    alignment_result, hits_at_k, mean_rank, mrr = greedy_alignment(
        embeds1=S_aligned.numpy(),
        embeds2=T_aligned.numpy(),
        top_k=[1, 5, 10, 50],
        threads_num=1,
        metric='inner',
        normalize=False
    )
    final_save_dir = args.output_dir
    os.makedirs(final_save_dir, exist_ok=True)

    torch.save(swa_model.state_dict(), os.path.join(
        final_save_dir, "final_alignment_model.pt"))

    with open(os.path.join(final_save_dir, "final_metadata.json"), "w") as f:
        json.dump({
            "final_loss": best_metrics["final_loss"] if best_metrics else None,
            "epochs": 500,
            "input_dim": input_dim,
            "hidden_dim": 256,
            "lr": 0.003,
            "weights": best_metrics["weights"] if best_metrics else {},
            "MRR": mrr,
            "Hits@1": hits_at_k[0],
            "Hits@5": hits_at_k[1],
            "Hits@10": hits_at_k[2],
            "Hits@50": hits_at_k[3],
            "best_trial_folder": best_trial_dir 
        }, f, indent=4)

    # Show results
    print(f"\nTest Evaluation Results:")
    print(f"    Hits@1:  {hits_at_k[0]:.4f}")
    print(f"    Hits@5:  {hits_at_k[1]:.4f}")
    print(f"    Hits@10: {hits_at_k[2]:.4f}")
    print(f"    Hits@50: {hits_at_k[3]:.4f}")
    print(f"    MR:      {mean_rank:.2f}")
    print(f"    MRR:     {mrr:.4f}")


def _one_thread(start, end, embeds1, embeds2, top_k, metric, normalize):
    result = []
    for i in range(start, end):
        e1 = embeds1[i]
        if normalize:
            e1 = e1 / np.linalg.norm(e1)

        sims = []
        for j in range(len(embeds2)):
            e2 = embeds2[j]
            if normalize:
                e2 = e2 / np.linalg.norm(e2)

            if metric == 'cosine':
                score = np.dot(e1, e2) / (np.linalg.norm(e1)
                                          * np.linalg.norm(e2))
            elif metric == 'euclidean':
                score = -np.linalg.norm(e1 - e2)
            else:  # inner product
                score = np.dot(e1, e2)

            sims.append((score, j))

        top_k_sim = heapq.nlargest(max(top_k), sims)
        result.append([idx for _, idx in top_k_sim])
    return result


if __name__ == "__main__":
    main()

import os
import gc
import time
import json
import torch
import shutil
import random
import logging
import pickle
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from rdflib import Graph
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from skopt import gp_minimize
from skopt.space import Real
from collections import OrderedDict
from dicee import KGE, intialize_model
from dicee.static_funcs import get_er_vocab, get_re_vocab
from dicee.eval_static_funcs import evaluate_link_prediction_performance

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Enable garbage collection and fix seed for reproducibility
gc.enable()
random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Neural Alignment Pipeline for KGE")
    parser.add_argument('-source_dir', type=str, required=True,
                        help="Path to the first dataset folder")
    parser.add_argument('--target_dir', type=str, required=True,
                        help="Path to the second dataset folder")
    parser.add_argument('--alignment_dir', type=str, required=True,
                        help="Path to folder with alignment files (.nt, .ttl, etc.)")
    parser.add_argument('--test_triples', type=str,
                        required=True, help="Path to the test triples file")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the results")
    return parser.parse_args()


def main(args):
    # --- Initialization ---
    directory_1 = args.dir1
    directory_2 = args.dir2
    alignment_dir = args.alignment_dir
    test_triples_path = args.test_triples
    output_dir = args.output_dir
    previous_model_folder = os.path.join(output_dir, "previous_model")
    new_model_folder = os.path.join(output_dir, "new_model")
    # define this BEFORE using it
    save_dir = os.path.join(output_dir, "aligned_embeddings")
    os.makedirs(save_dir, exist_ok=True)  # now safe to use

    previous_model_path = os.path.join(previous_model_folder, "model.pt")
    new_model_path = os.path.join(new_model_folder, "model.pt")
    save_dir = os.path.join(output_dir, "aligned_embeddings")

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
        """ Remove angle brackets, extra `>>`, and `<<` from URIs. """
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

        # Check if the folder exists
        if not os.path.exists(folder_path):
            logging.error(f"Folder '{folder_path}' does not exist.")
            return alignment_dict

        if not os.listdir(folder_path):  # Folder is empty
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

                        # Ensure predicate is owl:sameAs
                        if "sameAs" in pred:
                            alignment_dict[subj] = obj
                        else:
                            logging.warning(
                                f"Skipping triple with unexpected predicate: {subj} {pred} {obj}")

                elif extension in ['', '.txt']:
                    with open(file_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 2:
                                # Handle tab-separated TSV
                                subj = clean_uri(parts[0])
                                obj = clean_uri(parts[1])
                                alignment_dict[subj] = obj
                            elif len(parts) == 4 and "sameAs" in parts[1]:
                                # Handle sameAs line
                                subj = clean_uri(parts[0])
                                obj = clean_uri(parts[2])
                                alignment_dict[subj] = obj
                            else:
                                logging.warning(
                                    f"Skipping line (unexpected format): {line.strip()}")

                else:
                    logging.warning(f"Unsupported file type: {file_path}")
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")

        return alignment_dict

    alignment_dict = build_alignment_dict(alignment_dir)
    print(f"Number of entries in alignment_dict: {len(alignment_dict)}")

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

    cleaned_alignment_dict = clean_dict(alignment_dict)

    def create_train_test_matrices(alignment_dict, entity_embeddings1, entity_embeddings2, test_size=0.1):
        """Generates train and test matrices based on alignment dictionary."""
        filtered_alignment_dict = {
            k: v for k, v in alignment_dict.items() if k in entity_embeddings1.index and v in entity_embeddings2.index
        }

        if not filtered_alignment_dict:
            logging.warning(
                "No valid entries found in the alignment dictionary. Skipping train-test split.")
            return None, None, None, None

        # Perform train-test split
        train_ents, test_ents = train_test_split(
            list(filtered_alignment_dict.keys()), test_size=test_size, random_state=42)

        # Retrieve embeddings
        S_train = entity_embeddings1.loc[train_ents].values
        T_train = entity_embeddings2.loc[list(
            map(filtered_alignment_dict.get, train_ents))].values
        S_test = entity_embeddings1.loc[test_ents].values if test_ents else None
        T_test = entity_embeddings2.loc[list(
            map(filtered_alignment_dict.get, test_ents))].values if test_ents else None

        return S_train, T_train, S_test, T_test

    S_train, T_train, S_test, T_test = create_train_test_matrices(
        cleaned_alignment_dict, entity_embeddings1, entity_embeddings2, test_size=0.1)

    def normalize_and_scale(data, reference_data=None):
        """
        Normalize and scale data using its mean and standard deviation, or those of a reference dataset.

        Parameters:
        - data: ndarray, the data to be normalized.
        - reference_data: ndarray, the reference data used for calculating mean and scale. If None, uses `data` itself.

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

    # Example usage for S_train, T_train, S_test, and T_test
    # Assuming S_train, T_train, S_test, and T_test are NumPy arrays
    S_train_normalized, S_train_mean, S_train_scale = normalize_and_scale(
        S_train)
    T_train_normalized, T_train_mean, T_train_scale = normalize_and_scale(
        T_train)

    S_test = pd.DataFrame(np.array(S_test))
    T_test = pd.DataFrame(np.array(T_test))
    S_test_normalized, _, _ = normalize_and_scale(
        S_test, reference_data=S_train)
    T_test_normalized, _, _ = normalize_and_scale(
        T_test, reference_data=T_train)
    S_test_normalized = S_test_normalized.to_numpy() if hasattr(
        S_test_normalized, 'to_numpy') else S_test_normalized
    T_test_normalized = T_test_normalized.to_numpy() if hasattr(
        T_test_normalized, 'to_numpy') else T_test_normalized

    # Define the model

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

    def loss_fn(S_aligned, T_aligned, S_shared, T_shared, S_train_tensor, T_train_tensor, w1, w2, w3, w4):
        mse_loss = nn.MSELoss()

        structure_loss = mse_loss(
            S_aligned, S_train_tensor) + mse_loss(T_aligned, T_train_tensor)
        alignment_loss = mse_loss(S_shared, T_shared)
        cosine_sim_loss = mse_loss(nn.functional.cosine_similarity(
            S_train_tensor, S_aligned, dim=1), torch.ones_like(S_train_tensor[:, 0]))
        magnitude_loss = mse_loss(torch.norm(S_train_tensor, dim=1), torch.norm(S_aligned, dim=1)) + \
            mse_loss(torch.norm(T_train_tensor, dim=1),
                     torch.norm(T_aligned, dim=1))

        # Compute total weight sum (avoid division by zero)
        weight_sum = w1 + w2 + w3 + w4
        if weight_sum == 0:
            # Fallback if all weights are 0
            return structure_loss + alignment_loss + cosine_sim_loss + magnitude_loss

        # Normalize the loss values by the weight sum
        total_loss = (w1 * structure_loss + w2 * alignment_loss +
                      w3 * cosine_sim_loss + w4 * magnitude_loss) / weight_sum

        return total_loss

    # **Function to Train Alignment Model**

    def train_alignment_model(S_train, T_train, input_dim, hidden_dim, epochs, lr, w1, w2, w3, w4):
        model = SharedSpaceAlignmentNN(input_dim, hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        S_train_tensor = torch.tensor(S_train, dtype=torch.float32)
        T_train_tensor = torch.tensor(T_train, dtype=torch.float32)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()

            # Forward pass
            S_aligned, S_shared = model(S_train_tensor)
            T_aligned, T_shared = model(T_train_tensor)

            loss = loss_fn(S_aligned, T_aligned, S_shared, T_shared,
                           S_train_tensor, T_train_tensor, w1, w2, w3, w4)

            # Backpropagation
            loss.backward()
            optimizer.step()

        return model

    def load_json(p: str) -> dict:
        with open(p, 'r') as r:
            args = json.load(r)
        return args

    def initialize_models_and_update_embeddings(entity_embeddings, relation_embeddings, new_model_folder):
        """
        Initializes a KGE model using the config in `new_model_folder` and updates it with
        the given entity and relation embeddings.

        Args:
            entity_embeddings (pd.DataFrame or np.ndarray): Entity embeddings.
            relation_embeddings (pd.DataFrame or np.ndarray): Relation embeddings.
            new_model_folder (str): Path to the folder containing the configuration and report files.

        Returns:
            model: A KGE model initialized and updated with the provided embeddings.
        """

        def extract_parameters(entity_embeddings, relation_embeddings):
            num_entities = entity_embeddings.shape[0]
            num_relations = relation_embeddings.shape[0]
            embedding_dim = entity_embeddings.shape[1]
            return {
                "num_entities": num_entities,
                "num_relations": num_relations,
                "embedding_dim": embedding_dim
            }

        # Load config and report files
        config_path = os.path.join(new_model_folder, 'configuration.json')
        report_path = os.path.join(new_model_folder, 'report.json')

        configs = load_json(config_path)
        report = load_json(report_path)

        # Safely override critical parameters from data
        model_name = configs["model"]  # e.g., "TransE", "ComplEx", etc.
        print(f"Model from config: {model_name}")

        # Override num_entities, num_relations, and embedding_dim based on embeddings (not report!)
        params = extract_parameters(entity_embeddings, relation_embeddings)
        configs["num_entities"] = params["num_entities"]
        configs["num_relations"] = params["num_relations"]
        configs["embedding_dim"] = params["embedding_dim"]

        # Initialize model from config
        model, _ = intialize_model(configs)

        # Convert to torch tensors
        entity_tensor = torch.tensor(entity_embeddings, dtype=torch.float32)
        relation_tensor = torch.tensor(
            relation_embeddings, dtype=torch.float32)

        # Update model's embedding weights
        with torch.no_grad():
            model.entity_embeddings.weight.data = entity_tensor
            model.relation_embeddings.weight.data = relation_tensor
            model.entity_embeddings.weight.requires_grad = False
            model.relation_embeddings.weight.requires_grad = False

        return model

    @torch.no_grad()
    def evaluate_link_prediction_performance(model: KGE, triples, er_vocab, re_vocab):
        assert isinstance(model, KGE)
        model.model.eval()
        hits = dict()
        reciprocal_ranks = []
        num_entities = model.num_entities
        all_entities = torch.arange(0, num_entities).long()

        skipped = 0  # Count how many triples were skipped

        for i in tqdm(range(len(triples))):
            str_h, str_r, str_t = triples[i]

            try:
                h = model.get_entity_index(str_h)
                r = model.get_relation_index(str_r)
                t = model.get_entity_index(str_t)
            except KeyError:
                skipped += 1
                continue  # Skip this triple if any part is not found

            x_tail = torch.stack((torch.tensor(h).repeat(num_entities),
                                  torch.tensor(r).repeat(num_entities),
                                  all_entities), dim=1)
            predictions_tails = model.model.forward_triples(x_tail)

            x_head = torch.stack((all_entities,
                                  torch.tensor(r).repeat(num_entities),
                                  torch.tensor(t).repeat(num_entities)), dim=1)
            predictions_heads = model.model.forward_triples(x_head)

            filt_tails = [model.entity_to_idx[i] for i in er_vocab.get(
                (str_h, str_r), []) if i in model.entity_to_idx]
            target_tail_score = predictions_tails[t].item()
            predictions_tails[filt_tails] = -np.Inf
            predictions_tails[t] = target_tail_score
            rank_t = (predictions_tails.argsort(descending=True)
                      == t).nonzero(as_tuple=True)[0].item() + 1

            filt_heads = [model.entity_to_idx[i] for i in re_vocab.get(
                (str_r, str_t), []) if i in model.entity_to_idx]
            target_head_score = predictions_heads[h].item()
            predictions_heads[filt_heads] = -np.Inf
            predictions_heads[h] = target_head_score
            rank_h = (predictions_heads.argsort(descending=True)
                      == h).nonzero(as_tuple=True)[0].item() + 1

            reciprocal_ranks.append(1 / rank_t + 1 / rank_h)
            for k in [1, 3, 10]:
                hits.setdefault(k, []).append((rank_t <= k) + (rank_h <= k))

        total_evaluated = len(triples) - skipped
        if total_evaluated == 0:
            print("No valid triples were evaluated.")
            return {"H@1": 0.0, "H@3": 0.0, "H@10": 0.0, "MRR": 0.0}

        mrr = sum(reciprocal_ranks) / (2 * total_evaluated)
        results = {
            "H@1": sum(hits[1]) / (2 * total_evaluated),
            "H@3": sum(hits[3]) / (2 * total_evaluated),
            "H@10": sum(hits[10]) / (2 * total_evaluated),
            "MRR": mrr
        }

        print(f"Skipped {skipped} triples due to missing entities/relations.")
        return results

    def evaluate_model(kge_model, test_triples):
        if kge_model is None or test_triples is None:
            return -1  # No previous model

        er_vocab = get_er_vocab(test_triples)
        re_vocab = get_re_vocab(test_triples)

        performance = evaluate_link_prediction_performance(
            kge_model, test_triples, er_vocab, re_vocab)
        return performance['MRR']

    input_dim = S_train_normalized.shape[1]
    # **Prepare Full Dataset Embeddings**
    merged_embeddings_full = pd.concat(
        [entity_embeddings1, entity_embeddings2])

    # Check explicitly for duplicates before removing
    duplicates = merged_embeddings_full.index[merged_embeddings_full.index.duplicated(
        keep=False)]

    # Remove duplicates, clearly keeping the first occurrence only
    merged_embeddings_full_no_duplicates = merged_embeddings_full[~merged_embeddings_full.index.duplicated(
        keep='first')]

    entities_to_remove = set(cleaned_alignment_dict.values())

    # Now drop these entities from the dataframe
    final_embeddings_df = merged_embeddings_full_no_duplicates.drop(
        labels=entities_to_remove, errors='ignore'
    )

    sorted_merged_embeddings_full = final_embeddings_df.sort_index()
    # Display first 10 sorted entities
    print(sorted_merged_embeddings_full.head(10))
    merged_embeddings_normalized, _, _ = normalize_and_scale(
        sorted_merged_embeddings_full, reference_data=S_train)
    merged_embeddings_tensor = torch.tensor(
        merged_embeddings_normalized.values, dtype=torch.float32)

    # **Prepare Relation Embeddings**
    merged_relation = pd.concat([relation_embeddings1, relation_embeddings2])
    sorted_merged_relation_full = merged_relation.sort_index()
    sorted_merged_relation_full = sorted_merged_relation_full[~sorted_merged_relation_full.index.duplicated(
        keep='first')]
    merged_relation_normalized, _, _ = normalize_and_scale(
        sorted_merged_relation_full, reference_data=S_train)
    relation_tensor = torch.tensor(
        merged_relation_normalized.values, dtype=torch.float32)

    print(f" Merged entity embeddings shape: {merged_embeddings_tensor.shape}")
    print(f" Merged relation embeddings shape: {relation_tensor.shape}")

    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

    # Define save paths
    entity_save_path = os.path.join(save_dir, "merged_entity_embeddings.pt")
    relation_save_path = os.path.join(
        save_dir, "merged_relation_embeddings.pt")

    # Convert to PyTorch tensors (if not already)
    merged_entity_tensor = torch.tensor(
        merged_embeddings_tensor.numpy(), dtype=torch.float32)
    relation_tensor = torch.tensor(
        merged_relation_normalized.values, dtype=torch.float32)

    # Save tensors
    torch.save(merged_entity_tensor, entity_save_path)
    torch.save(relation_tensor, relation_save_path)

    print(f"Merged entity embeddings saved at: {entity_save_path}")
    print(f"Merged relation embeddings saved at: {relation_save_path}")

    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

    # Convert to PyTorch tensors (if not already)
    merged_entity_tensor = torch.tensor(
        merged_embeddings_tensor.numpy(), dtype=torch.float32)
    relation_tensor = torch.tensor(
        merged_relation_normalized.values, dtype=torch.float32)

    os.makedirs(previous_model_folder, exist_ok=True)
    os.makedirs(new_model_folder, exist_ok=True)

    previous_model_path = os.path.join(previous_model_folder, "model.pt")
    new_model_path = os.path.join(new_model_folder, "model.pt")

    # **Load Test Triples**
    test_triples = pd.read_csv(test_triples_path, sep="\s+", header=None, names=[
                               'subject', 'relation', 'object'], dtype=str).values.tolist()[:1000]

    if not os.path.exists(previous_model_path):
        raise FileNotFoundError(
            f"Expected previous model at {previous_model_path} not found.")
    if not os.path.exists(new_model_path):
        raise FileNotFoundError(
            f"Expected new model at {new_model_path} not found.")

    print(f"Found previous model in {previous_model_folder}, loading it...")
    pre_trained_kge = KGE(path=previous_model_folder)
    best_mrr = evaluate_model(pre_trained_kge, test_triples)

    # **Grid Search for Best Alignment Model Based on MRR**
    #  Track start time for Grid Search
    grid_search_start_time = time.time()

    input_dim = S_train.shape[1]
    best_model = None

    param_grid = {
        "w1": np.linspace(0, 1, 1),
        "w2": np.linspace(0, 1, 1),
        "w3": np.linspace(0, 0.5, 1),
        "w4": np.linspace(0, 0.5, 1)
    }

    # **Grid Search Process**
    for idx, (w1, w2, w3, w4) in enumerate(product(*param_grid.values()), 1):
        print(f"\n🔹 [{idx}] NAAS with w1={w1}, w2={w2}, w3={w3}, w4={w4}")

        # **Step 1: Train Alignment Model**
        alignment_model = train_alignment_model(
            S_train, T_train, input_dim, 256, 500, 0.001, w1, w2, w3, w4)

        # **Step 2: Apply Alignment to All Embeddings**
        with torch.no_grad():
            aligned_entity_embeddings, _ = alignment_model(
                merged_embeddings_tensor.clone().detach())
            aligned_entity_embeddings = aligned_entity_embeddings.numpy()

            aligned_relation_embeddings, _ = alignment_model(
                relation_tensor.clone().detach())
            aligned_relation_embeddings = aligned_relation_embeddings.numpy()

        # **Step 3: Train New KGE Model on Aligned Embeddings**
        kge_model = initialize_models_and_update_embeddings(
            aligned_entity_embeddings, aligned_relation_embeddings, new_model_folder)

        #  Save the model inside the folder (Torch format)
        torch.save(kge_model.state_dict(), os.path.join(
            new_model_folder, "model.pt"))

        #  Now, Load KGE from Folder (So Evaluation Works)
        new_trained_kge = KGE(path=new_model_folder)

        # Step 4: Evaluate New Model on Link Prediction
        new_mrr = evaluate_model(new_trained_kge, test_triples)

        # Step 5: Compare and Save Best Model
        if new_mrr > best_mrr:
            best_mrr = new_mrr
            # Store best Grid Search parameters
            best_grid_params = [w1, w2, w3, w4]
            print(
                f" New model (MRR={new_mrr}) is better. Replacing previous model.")

            # Remove old previous model (if exists)
            if os.path.exists(previous_model_path):
                os.remove(previous_model_path)

            # Move new model inside previous_model_folder
            shutil.move(new_model_path, previous_model_path)
        else:
            print(
                f" New model (MRR={new_mrr}) is NOT better. Keeping old model.")

    print(f" Best model is inside {previous_model_folder}")

    #  Compute and print total Grid Search execution time
    grid_search_end_time = time.time()
    grid_search_duration = grid_search_end_time - grid_search_start_time
    print(
        f" Grid Search took {grid_search_duration:.2f} seconds ({grid_search_duration / 60:.2f} minutes)")

    ###### BAYESIAN#####
    bayesian_start_time = time.time()

    # We only optimize w1, w2, w3, and enforce w4 = 1 - (w1 + w2 + w3)
    search_space = [Real(0, 1), Real(0, 1), Real(0, 1)]

    def objective(params):
        """
        Objective function for Bayesian Optimization.
        This function trains the alignment model with given weights and evaluates MRR.
        """
        w1, w2, w3 = params
        total = w1 + w2 + w3
        if total > 1:
            w1 /= total
            w2 /= total
            w3 /= total
        w4 = 1 - (w1 + w2 + w3)  # Ensures w1 + w2 + w3 + w4 = 1

        print(
            f"\n [Bayesian] Training alignment model with w1={w1}, w2={w2}, w3={w3}, w4={w4}...")

        # **Train Alignment Model**
        alignment_model = train_alignment_model(
            S_train, T_train, input_dim, 256, 500, 0.001, w1, w2, w3, w4)

        # **Step 2: Apply Alignment to Embeddings**
        with torch.no_grad():
            aligned_entity_embeddings, _ = alignment_model(
                merged_embeddings_tensor.clone().detach())
            aligned_entity_embeddings = aligned_entity_embeddings.numpy()

            aligned_relation_embeddings, _ = alignment_model(
                relation_tensor.clone().detach())
            aligned_relation_embeddings = aligned_relation_embeddings.numpy()

        # **Step 3: Train New KGE Model on Aligned Embeddings**
        kge_model = initialize_models_and_update_embeddings(
            aligned_entity_embeddings, aligned_relation_embeddings, new_model_folder)

        #  Save the new trained model inside `new_model_folder`
        torch.save(kge_model.state_dict(), os.path.join(
            new_model_folder, "model.pt"))

        #  Now, Load the KGE model from `new_model_folder` for proper evaluation (Same as Grid Search)
        new_trained_kge = KGE(path=new_model_folder)

        # Step 4: Evaluate New Model on Link Prediction
        new_mrr = evaluate_model(new_trained_kge, test_triples)

        print(f" [Bayesian] Model MRR={new_mrr}")

        # **Return negative MRR for minimization**
        return -new_mrr

    # **Run Bayesian Optimization**
    print("\n🔍 Running Bayesian Optimization...")
    bayes_results = gp_minimize(objective, search_space, n_calls=10)

    # **Get Best Bayesian Parameters**
    best_bayesian_params = bayes_results.x
    w1, w2, w3 = best_bayesian_params
    w4 = max(0, 1 - (w1 + w2 + w3))  # Ensure w4 is non-negative
    print(f"Best Bayesian Parameters: w1={w1}, w2={w2}, w3={w3}, w4={w4}")

    # **Train Final Model with Best Bayesian Parameters**
    best_bayesian_model = train_alignment_model(
        S_train, T_train, input_dim, 256, 500, 0.001, w1, w2, w3, w4)

    # **Apply Alignment to Full Dataset**
    with torch.no_grad():
        aligned_embeddings_bayesian, _ = best_bayesian_model(
            merged_embeddings_tensor)
        aligned_relation_bayesian, _ = best_bayesian_model(relation_tensor)

    entity_embeddings_bayesian = aligned_embeddings_bayesian.numpy()
    relation_embeddings_bayesian = aligned_relation_bayesian.numpy()

    # **Train Final KGE Model**
    final_bayesian_kge = initialize_models_and_update_embeddings(
        aligned_entity_embeddings, aligned_relation_embeddings, new_model_folder)

    #  Ensure `new_model_folder` exists
    os.makedirs(new_model_folder, exist_ok=True)

    # Save the final Bayesian-trained model inside `new_model_folder`
    torch.save(final_bayesian_kge.state_dict(),
               os.path.join(new_model_folder, "model.pt"))

    # Load Bayesian Model from `new_model_folder` for final evaluation (Matching Grid Search)
    final_bayesian_trained_kge = KGE(path=new_model_folder)

    # **Evaluate Bayesian Model**
    # Load full test set for final evaluation
    full_test_triples = pd.read_csv(test_triples_path, sep="\s+", header=None, names=[
                                    'subject', 'relation', 'object'], dtype=str).values.tolist()

    # Evaluate again on full test set with all metrics
    er_vocab_full = get_er_vocab(full_test_triples)
    re_vocab_full = get_re_vocab(full_test_triples)
    final_metrics = evaluate_link_prediction_performance(
        final_bayesian_trained_kge, full_test_triples, er_vocab_full, re_vocab_full)

    print(
        f" Final Bayesian Model Performance on FULL test set:\n{final_metrics}")

    # **Compare and Save the Best Model (Same as Grid Search)**
    best_mrr_bayesian = -bayes_results.fun  # Convert back to positive MRR
    if best_mrr_bayesian > best_mrr:
        print(
            f" Bayesian Optimization found a better model (MRR={best_mrr_bayesian})! Replacing Grid Search model.")

        # Remove old best model
        if os.path.exists(previous_model_path):
            os.remove(previous_model_path)

        # Move Bayesian Model to `previous_model_folder` (Same as Grid Search)
        shutil.move(os.path.join(new_model_folder,
                    "model.pt"), previous_model_path)
        print(
            f"Best model saved in {previous_model_folder} using Bayesian Optimization.")
    else:
        print(
            f" Grid Search model (MRR={best_mrr}) is still the best. Keeping it.")

    # **Final Results**
    print(f" Best Grid Search Model MRR: {best_mrr}")
    print(f" Best Bayesian Model MRR: {best_mrr_bayesian}")

    #  Compute and print total Bayesian Optimization execution time
    bayesian_end_time = time.time()
    bayesian_duration = bayesian_end_time - bayesian_start_time
    print(
        f" Bayesian Optimization took {bayesian_duration:.2f} seconds ({bayesian_duration / 60:.2f} minutes)")

    # Save Best Parameters with full metrics
    best_params = {
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "w4": w4,
        "MRR": final_metrics["MRR"],
        "H@1": final_metrics["H@1"],
        "H@3": final_metrics["H@3"],
        "H@10": final_metrics["H@10"]
    }

    with open(os.path.join(previous_model_folder, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    print(
        f" Best parameters and scores saved in {previous_model_folder}/best_params.json")


if __name__ == "__main__":
    args = parse_args()
    main(args)

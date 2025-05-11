"""
Procrustes Alignment Script

This script aligns embeddings from two datasets into a shared vector space using Orthogonal Procrustes analysis. 
It supports downstream tasks such as link prediction, graph integration, and entity resolution.

Key Features:
- Loads pre-trained entity and relation embeddings from two datasets.
- Parses `sameAs` alignment links from LIMES or manually curated files.
- Trains Procrustes alignment in both directions (source→target and target→source).
- Merges aligned and unaligned embeddings, preserving entity identity.
- Saves aligned embeddings and models in PyTorch format.

Input:
- --embedding_folder: Root folder containing two dataset subfolders with model.pt and mappings.
- --alignment_dict_path: Path to manually curated alignment files.
- --alignmentlimes_dict_path: Path to LIMES-based alignment files.
- --output_folder: Path to save aligned models and mappings.

Designed for modular use in knowledge graph integration pipelines.
"""

# Import external libraries
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.linalg import orthogonal_procrustes

# Import standard Python libraries
import os
import pickle
import argparse
import logging
from rdflib import Graph
from collections import OrderedDict
from pathlib import Path
from dicee.static_funcs import get_er_vocab, get_re_vocab
from link_prediction import evaluate_link_prediction_performance  # Use your improved version


# Import project-specific modules
from dicee import KGE, Keci, TransE, intialize_model

# Set up logging for better error handling
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Run Procrustes alignment on two embedding sets.")
    parser.add_argument("--embedding_folder", required=True, help="Path to embedding folder with two dataset subfolders.")
    parser.add_argument("--alignment_dict_path", required=True, help="Path to folder with pre-aligned sameAs links.")
    parser.add_argument("--alignmentlimes_dict_path", required=True, help="Path to folder with LIMES alignment links.")
    parser.add_argument("--output_folder", required=True, help="Path to output folder to save aligned embeddings and models.")
    return parser.parse_args()



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

    # Check for both .p (pickle) and .csv versions of entity_to_idx and relation_to_idx
    entity_to_id_p = os.path.join(directory, "entity_to_idx.p")
    entity_to_id_csv = os.path.join(directory, "entity_to_idx.csv")
    
    relation_to_id_p = os.path.join(directory, "relation_to_idx.p")
    relation_to_id_csv = os.path.join(directory, "relation_to_idx.csv")

    # Determine which entity_to_idx file exists
    if os.path.exists(entity_to_id_p):
        entity_to_id_path = entity_to_id_p
    elif os.path.exists(entity_to_id_csv):
        entity_to_id_path = entity_to_id_csv
    else:
        logging.error(f"Entity-to-ID file not found in {directory}")
        raise FileNotFoundError(f"Missing entity_to_idx file: {entity_to_id_p} or {entity_to_id_csv}")

    # Determine which relation_to_idx file exists
    if os.path.exists(relation_to_id_p):
        relation_to_id_path = relation_to_id_p
    elif os.path.exists(relation_to_id_csv):
        relation_to_id_path = relation_to_id_csv
    else:
        logging.error(f"Relation-to-ID file not found in {directory}")
        raise FileNotFoundError(f"Missing relation_to_idx file: {relation_to_id_p} or {relation_to_id_csv}")

    logging.debug(f"Files found: {model_path}, {entity_to_id_path}, {relation_to_id_path}")
    return model_path, entity_to_id_path, relation_to_id_path


def load_embeddings(model_path, entity_to_id_path, relation_to_id_path):
    """Load embeddings and mappings for entities and relations."""
    
    logging.info(f"Loading model weights from: {model_path}")
    model_weights = torch.load(model_path, map_location='cpu')

    # Load entity and relation embeddings
    entity_embeddings = model_weights['entity_embeddings.weight'].cpu().detach().numpy()
    relation_embeddings = model_weights['relation_embeddings.weight'].cpu().detach().numpy()

    # Handle entity-to-ID mapping
    if entity_to_id_path.endswith('.csv'):
        logging.info(f"Loading entity mappings from CSV: {entity_to_id_path}")
        
        # Read CSV correctly
        entity_df = pd.read_csv(entity_to_id_path, header=None, names=["index", "entity"])
        
        # Convert index → URI dictionary
        entity_to_id = dict(zip(entity_df["index"], entity_df["entity"]))

    elif entity_to_id_path.endswith(('.p', '.pkl')):  
        logging.info(f"Loading entity mappings from Pickle: {entity_to_id_path}")
        with open(entity_to_id_path, 'rb') as f:
            entity_to_id = pickle.load(f)
    
    else:
        raise ValueError(f"Unsupported file format: {entity_to_id_path}")

    # Handle relation-to-ID mapping
    if relation_to_id_path.endswith('.csv'):
        logging.info(f"Loading relation mappings from CSV: {relation_to_id_path}")
        
        relation_df = pd.read_csv(relation_to_id_path, header=None, names=["index", "relation"])
        relation_to_id = dict(zip(relation_df["index"], relation_df["relation"]))

    elif relation_to_id_path.endswith(('.p', '.pkl')):  
        logging.info(f"Loading relation mappings from Pickle: {relation_to_id_path}")
        with open(relation_to_id_path, 'rb') as f:
            relation_to_id = pickle.load(f)
    
    else:
        raise ValueError(f"Unsupported file format: {relation_to_id_path}")

    # Match embeddings to URIs using entity index
    sorted_entities = [entity_to_id[i] for i in range(len(entity_embeddings))]
    sorted_relations = [relation_to_id[i] for i in range(len(relation_embeddings))]

    # Ensure lengths match
    if len(entity_embeddings) != len(sorted_entities):
        logging.warning(f"Mismatch: {len(entity_embeddings)} entity embeddings vs. {len(sorted_entities)} entities. Fixing it...")
        sorted_entities = sorted_entities[:len(entity_embeddings)]
    
    if len(relation_embeddings) != len(sorted_relations):
        logging.warning(f"Mismatch: {len(relation_embeddings)} relation embeddings vs. {len(sorted_relations)} relations. Fixing it...")
        sorted_relations = sorted_relations[:len(relation_embeddings)]

    # Create DataFrames
    entity_embeddings_df = pd.DataFrame(entity_embeddings, index=sorted_entities)
    relation_embeddings_df = pd.DataFrame(relation_embeddings, index=sorted_relations)

    return entity_embeddings_df, relation_embeddings_df



def remove_brackets_from_indices(embeddings_df):
    """Remove < and > from each index in the DataFrame."""
    cleaned_index = [str(uri).strip('<>') for uri in embeddings_df.index]  # Convert to string before stripping
    embeddings_df.index = cleaned_index
    return embeddings_df


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
                g.parse(file_path, format='nt' if extension == '.nt' else 'ttl')
                for subj, _, obj in g:  # Ignore the predicate
                    alignment_dict[str(subj).strip('<>')] = str(obj).strip('<>')
            elif extension in ['', '.txt']:
                # Parse text-based alignment files
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            entity1, entity2 = parts
                            alignment_dict[entity1.strip('<>')] = entity2.strip('<>')
                        #else:
                            #logging.warning(f"Skipping line (unexpected format): {line.strip()}")       
            else:
                logging.warning(f"Unsupported file type: {file_path}")
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")

    return alignment_dict


def extract_original_alignment_with_indices(d1_entity_to_idx, d2_entity_to_idx, original_alignment_dict):
    # Initialize an empty dictionary to store the new alignment with indices
    original_with_indices = {}

    # Iterate over the original alignment dictionary
    for d1_uri, d2_uri in original_alignment_dict.items():
        # Check if the URI exists in both mappings
        if d1_uri in d1_entity_to_idx:
            d1_index = d1_entity_to_idx[d1_uri]
        else:
            print(f"URI not found in D1 entity mapping: {d1_uri}")
            continue  # Skip this entry if D1 URI is not found

        if d2_uri in d2_entity_to_idx:
            d2_index = d2_entity_to_idx[d2_uri]
        else:
            print(f"URI not found in D2 entity mapping: {d2_uri}")
            continue  # Skip this entry if D2 URI is not found

        # Add the data to the dictionary
        original_with_indices[d1_index] = (d1_uri, d2_index, d2_uri)

    # Print the first 5 entries
    print("\nFirst 5 entries of the original alignment with indices:")
    for d1_index, (d1_uri, d2_index, d2_uri) in list(original_with_indices.items())[:5]:
        print(
            f"D1 Index: {d1_index} (URI: {d1_uri}) -> D2 Index: {d2_index} (URI: {d2_uri})")

    return original_with_indices


def clean_dict(input_dict):
    """
    Cleans a dictionary by removing angle brackets and any trailing characters like '> .'
    from both keys and values.
    """
    return {
        k.strip('<>'): v.strip('<>') if isinstance(v, str) else v
        for k, v in input_dict.items()
    }


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


def normalize_and_scale(data, reference_data=None):
    """Normalize and scale data using its mean and standard deviation, or those of a reference dataset."""
    if reference_data is None:
        reference_data = data

    mean = reference_data.mean(axis=0)
    scale = np.sqrt(((reference_data - mean) ** 2).sum() /
                    reference_data.shape[0])
    normalized_data = (data - mean) / scale

    return normalized_data, mean, scale


def apply_procrustes_both_directions(S_train, T_train, S_test=None, T_test=None, reverse=False):
    """
    Performs Procrustes alignment for embeddings in both forward and reverse directions.

    Args:
        S_train (np.ndarray): Source training embeddings.
        T_train (np.ndarray): Target training embeddings.
        S_test (np.ndarray, optional): Source test embeddings.
        T_test (np.ndarray, optional): Target test embeddings.
        reverse (bool, optional): If True, aligns T to S. Defaults to False.

    Returns:
        tuple: Scaled and aligned embeddings for train and test sets, along with the rotation matrix (R).
    """
    # Normalize and scale training embeddings
    scaled_S_train, mean_S, scale_S = normalize_and_scale(S_train)
    scaled_T_train, mean_T, scale_T = normalize_and_scale(T_train)

    # Compute Procrustes alignment
    if not reverse:
        R, _ = orthogonal_procrustes(scaled_S_train, scaled_T_train)
    else:
        R, _ = orthogonal_procrustes(scaled_T_train, scaled_S_train)

    # Normalize and scale test embeddings if provided
    scaled_S_test = None
    scaled_T_test = None
    if S_test is not None and T_test is not None:
        scaled_S_test, _, _ = normalize_and_scale(
            S_test, reference_data=S_train)
        scaled_T_test, _, _ = normalize_and_scale(
            T_test, reference_data=T_train)

    if not reverse:
        S_test_aligned = scaled_S_test @ R if scaled_S_test is not None else None
        S_train_aligned = scaled_S_train @ R
        return scaled_S_train, S_train_aligned, scaled_T_train, S_test_aligned, scaled_S_test, scaled_T_test, R
    else:
        T_test_aligned = scaled_T_test @ R if scaled_T_test is not None else None
        T_train_aligned = scaled_T_train @ R
        return scaled_T_train, T_train_aligned, scaled_S_train, T_test_aligned, scaled_T_test, scaled_S_test, R



def extract_non_aligned_embeddings(
    alignment_dict, embeddings1, embeddings2,
    S_train=None, T_train=None, use_test_scaling=False,
    S_test=None, T_test=None, relation_embeddings1=None, relation_embeddings2=None,
    output_folder=None
):
    """
    Extracts and normalizes embeddings of non-aligned entities or relations, saves T_normalized indices.

    Args:
        alignment_dict (dict): Alignment dictionary for entities (use None for relations).
        embeddings1 (pd.DataFrame): Embeddings for the first dataset (entities or relations).
        embeddings2 (pd.DataFrame): Embeddings for the second dataset (entities or relations).
        S_train (np.ndarray): Training embeddings for dataset 1.
        T_train (np.ndarray): Training embeddings for dataset 2.
        use_test_scaling (bool): Whether to use test statistics for scaling.
        S_test (np.ndarray): Test embeddings for dataset 1.
        T_test (np.ndarray): Test embeddings for dataset 2.
        relation_embeddings1 (pd.DataFrame): Relation embeddings for dataset 1.
        relation_embeddings2 (pd.DataFrame): Relation embeddings for dataset 2.
        output_folder (str): Path to save the normalized indices as CSV.

    Returns:
        tuple: Normalized and optionally transformed embeddings for entities and relations.
            - (S_normalized, S_full_normalized, T_normalized, Sr_normalized, Tr_normalized)
    """
    if S_train is None or T_train is None:
        raise ValueError("Training data (S_train and T_train) must be provided.")

    # Use training or test statistics for normalization
    reference_data_S = S_test if use_test_scaling and S_test is not None else S_train
    reference_data_T = T_test if use_test_scaling and T_test is not None else T_train

    _, mean_S, scale_S = normalize_and_scale(S_train, reference_data=reference_data_S)
    _, mean_T, scale_T = normalize_and_scale(T_train, reference_data=reference_data_T)

    # Identify non-aligned entities
    if alignment_dict:
        embeddings1_non_aligned = list(set(embeddings1.index) - set(alignment_dict.keys()))
        embeddings2_non_aligned = list(set(embeddings2.index) - set(alignment_dict.values()))
    else:
        embeddings1_non_aligned = sorted(embeddings1.index)
        embeddings2_non_aligned = sorted(embeddings2.index)
        
    embeddings1_indices = sorted(set(embeddings1.index))

    S_non_aligned = embeddings1.loc[embeddings1_non_aligned].values
    T_non_aligned = embeddings2.loc[embeddings2_non_aligned].values
    S_full_aligned = embeddings1.loc[embeddings1_indices].values

    # Normalize entity embeddings
    S_normalized = (S_non_aligned - mean_S) / scale_S
    T_normalized = (T_non_aligned - mean_T) / scale_T
    S_full_normalized = (S_full_aligned - mean_S) / scale_S

    # Map non-aligned embeddings to their indices (URIs)
    T_normalized_indices = embeddings2_non_aligned

    # For relations (if provided)
    Sr_normalized, Tr_normalized = None, None
    if relation_embeddings1 is not None and relation_embeddings2 is not None:
        Sr_normalized = (relation_embeddings1.values - mean_S) / scale_S
        Tr_normalized = (relation_embeddings2.values - mean_T) / scale_T

    return S_normalized, S_full_normalized, T_normalized, Sr_normalized, Tr_normalized,T_normalized_indices



def filter_and_print_non_aligned_embeddings(alignment_dict, embeddings1, embeddings2):
    """
    Filters out aligned embeddings, sorts non-aligned indices, and prints statistics.

    Args:
        alignment_dict (dict): Alignment dictionary mapping entity URIs from dataset1 to dataset2.
        embeddings1 (pd.DataFrame): Embeddings for the first dataset's entities.
        embeddings2 (pd.DataFrame): Embeddings for the second dataset's entities.

    Returns:
        tuple: Two sorted lists containing non-aligned indices for dataset1 and dataset2 respectively.
    """
    if alignment_dict:
        # Ensure alignment_dict keys and values are strings
        alignment_keys = set(str(k) for k in alignment_dict.keys())
        alignment_values = set(str(v) for v in alignment_dict.values())

        # Filter and sort non-aligned indices
        embeddings1_non_aligned = sorted(
            list(set(embeddings1.index.astype(str)) - alignment_keys))
        embeddings2_non_aligned = sorted(
            list(set(embeddings2.index.astype(str)) - alignment_values))

    else:
        # If no alignment_dict is provided, consider all embeddings as non-aligned
        embeddings1_non_aligned = sorted(
            embeddings1.index.astype(str).tolist())
        embeddings2_non_aligned = sorted(
            embeddings2.index.astype(str).tolist())

    return embeddings1_non_aligned, embeddings2_non_aligned


def merge_embeddings(S_normalized, S_full_normalized, T_normalized, R, R_reverse,
                     S_train_aligned, S_test_aligned,
                     scaled_S_train, scaled_S_test,
                     T_train_aligned, T_test_aligned,
                     scaled_T_train, scaled_T_test, alignment_dict):

    logging.basicConfig(level=logging.DEBUG)
    logging.info("Starting merge_embeddings process.")

    # Step 1: Transform non-aligned embeddings
    S_na_transformed = S_normalized @ R
    S_full_norm_trans = S_full_normalized @ R
    T_na_transformed = T_normalized @ R_reverse

    # Step 2: Combine aligned embeddings
    S_aligned_full = np.concatenate((S_train_aligned, S_test_aligned), axis=0)
    S_nonaligned_full = np.concatenate((scaled_S_train, scaled_S_test), axis=0)
    T_aligned_full = np.concatenate((T_train_aligned, T_test_aligned), axis=0)
    T_nonaligned_full = np.concatenate((scaled_T_train, scaled_T_test), axis=0)


    averaged_embeddings_full = ((S_aligned_full + T_nonaligned_full) / 2 +
                                (T_aligned_full + S_nonaligned_full) / 2) / 2
    logging.debug(
        f"averaged_embeddings_full shape: {averaged_embeddings_full.shape}")

    # Combine all embeddings
    S_combined = (S_na_transformed + S_normalized) / 2
    S_combined_full = (S_full_norm_trans + S_full_normalized)/2
    T_combined = (T_na_transformed + T_normalized) / 2
    return S_combined, S_combined_full, T_combined, averaged_embeddings_full


def transform_merge_relation_embeddings(
    Sr_normalized , Tr_normalized, R, R_reverse, relation_embeddings1, relation_embeddings2
):
    """Transforms and merges relation embeddings using rotation matrices."""
    relations_1_transformed =  Sr_normalized @ R  # Apply R to Dataset 1 relations
    # Apply R_reverse to Dataset 2 relations
    relations_2_transformed = Tr_normalized @ R_reverse

    S_merged_relations = (relations_1_transformed + Sr_normalized)/2
    T_merged_relations = (relations_2_transformed + Tr_normalized)/2
    # Create DataFrames with the same indices as the original embeddings
    S_merged_df = pd.DataFrame(S_merged_relations, index=relation_embeddings1.index)
    T_merged_df = pd.DataFrame(T_merged_relations, index=relation_embeddings2.index)

    # Concatenate the merged DataFrames
    all_relation_embeddings= pd.concat([S_merged_df, T_merged_df])

    # Sort the concatenated DataFrame by index (URI)
    all_relation_embeddings_df = all_relation_embeddings.sort_index()
    
    # Remove duplicate indices while keeping the first occurrence
    all_relation_embeddings_df = all_relation_embeddings_df[~all_relation_embeddings_df.index.duplicated(keep='first')]

    # Convert the sorted DataFrame to a NumPy array without indices
    all_relation_embeddings_array = all_relation_embeddings_df.to_numpy()
    
    return all_relation_embeddings_df, all_relation_embeddings_array

def load_mapping(file_path):
    """Loads entity or relation mappings from either a pickle or CSV file."""
    try:
        if file_path.endswith(".p"):
            with open(file_path, "rb") as file:
                mapping = pickle.load(file)
        elif file_path.endswith(".csv"):
            mapping_df = pd.read_csv(file_path, header=None, names=["key", "value"])
            mapping = dict(zip(mapping_df["key"], mapping_df["value"]))
        else:
            raise ValueError("Unsupported file format. Use .p (pickle) or .csv")
        
        print(f"Loaded {len(mapping)} entries from {file_path}")
        return mapping
    except Exception as e:
        print(f"Error loading mapping from {file_path}: {e}")
        return None
    

def generate_train_set_with_uris(entity_to_idx_path, relation_to_idx_path, train_set_path, output_path):
    print("Starting generate_train_set_with_uris...")
    
    # Load entity and relation mappings
    entity_to_idx = load_mapping(entity_to_idx_path)
    relation_to_idx = load_mapping(relation_to_idx_path)
    
    if entity_to_idx is None or relation_to_idx is None:
        print("Failed to load mappings. Exiting...")
        return

    # Fix: Ensure CSV and Pickle formats match
    if entity_to_idx_path.endswith(".csv"):
        idx_to_entity = entity_to_idx  # CSV is already index → URI
    else:
        idx_to_entity = {v: k for k, v in entity_to_idx.items()}  # Pickle is URI → index

    if relation_to_idx_path.endswith(".csv"):
        idx_to_relation = relation_to_idx  # CSV is already index → relation
    else:
        idx_to_relation = {v: k for k, v in relation_to_idx.items()}  # Pickle is relation → index
    

    # Load train set
    try:
        train_data = np.load(train_set_path, allow_pickle=True)
        print(f" Loaded {len(train_data)} triples from {train_set_path}")
    except Exception as e:
        print(f"Error loading train.npy: {e}")
        return
    
    # Convert indices to URIs
    train_with_uris = []
    for triple in train_data:
        subject = idx_to_entity.get(triple[0], f"Unknown URI ({triple[0]})")
        predicate = idx_to_relation.get(triple[1], f"Unknown Relation ({triple[1]})")
        obj = idx_to_entity.get(triple[2], f"Unknown URI ({triple[2]})")
        train_with_uris.append([subject, predicate, obj])
    
    print(f"Processed {len(train_with_uris)} triples")
    
    # Save as numpy file
    try:
        np.save(output_path, np.array(train_with_uris, dtype=object))
        print(f"train_set_with_relation.npy saved to {output_path}")
    except Exception as e:
        print(f"Error saving train_set_with_relation.npy: {e}")



def generate_train_set_with_relation_indices(train_set_with_uris_path, uri_to_index, output_path):
    train_set_with_uris = np.load(train_set_with_uris_path, allow_pickle=True)
    updated_train_set = [
        [triple[0], uri_to_index.get(triple[1], triple[1]), triple[2]]
        for triple in train_set_with_uris
    ]
    np.save(output_path, np.array(updated_train_set, dtype=object))
    print(f"train_set_with_relation_indices.npy saved to {output_path}")


def generate_train_set_with_relation_uris(train_set_with_relation_indices_path, reversed_mapping, relation_index_to_uri, output_path):
    train_set = np.load(
        train_set_with_relation_indices_path, allow_pickle=True)
    revised_train_set = [
        (
            triple[0],
            relation_index_to_uri[triple[1]] if 0 <= triple[1] < len(relation_index_to_uri)
            else f"Unknown_relation_{triple[1]}",
            triple[2],
        )
        for triple in train_set
    ]
    np.save(output_path, np.array(revised_train_set, dtype=object))
    print(f"train_set_with_relation_uris.npy saved to {output_path}")


# Model Initialization
def extract_model_parameters(entity_embeddings, relation_embeddings):
    """Extracts dynamic model parameters from the given embeddings."""
    if entity_embeddings.shape[1] != relation_embeddings.shape[1]:
        raise ValueError(
            "Entity and relation embeddings must have the same embedding dimension.")

    num_entities = entity_embeddings.shape[0]
    num_relations = relation_embeddings.shape[0]
    embedding_dim = entity_embeddings.shape[1]

    return {
        "num_entities": num_entities,
        "num_relations": num_relations,
        "embedding_dim": embedding_dim
    }


def initialize_models_and_update_embeddings(
    entity_embeddings, relation_embeddings
):
    """
    Initializes two models using the same entity and relation embeddings, 
    but with different q_coefficients values.
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

    # Extract shared parameters
    params = extract_parameters(entity_embeddings, relation_embeddings)

    # Initialize the first model
    model, _ = intialize_model(args={
        "model": "TransE",
        "num_entities": params["num_entities"],
        "num_relations": params["num_relations"],
        "embedding_dim": params["embedding_dim"]
    })


    # Convert embeddings and q values to tensors
    entity_matrix = torch.tensor(entity_embeddings, dtype=torch.float32)
    relation_matrix = torch.tensor(relation_embeddings, dtype=torch.float32)

    # Update embeddings for both models
    with torch.no_grad():
        # Model 1
        model.entity_embeddings.weight.data = entity_matrix
        model.relation_embeddings.weight.data = relation_matrix

        # Disable gradients for model 1 embeddings
        model.entity_embeddings.weight.requires_grad = False
        model.relation_embeddings.weight.requires_grad = False

    return model




def procrustes_alignment_pipeline(
    alignment_dict, entity_embeddings1, entity_embeddings2, relation_embeddings1,
    relation_embeddings2, output_folder, alignment_type, cleaned_pre_aligned_dict,
    new_model_folder, previous_model_folder
):

    """
    Handles Procrustes alignment for a given alignment dictionary.

    Args:
        alignment_dict (dict): Alignment dictionary.
        entity_embeddings1 (pd.DataFrame): Entity embeddings for the first dataset.
        entity_embeddings2 (pd.DataFrame): Entity embeddings for the second dataset.
        relation_embeddings1 (pd.DataFrame): Relation embeddings for the first dataset.
        relation_embeddings2 (pd.DataFrame): Relation embeddings for the second dataset.
        output_folder (str): Directory to save aligned models and embeddings.
        alignment_type (str): Type of alignment, e.g., "pre_aligned" or "limes".

    Returns:
        None
    """
    print(f"Starting Procrustes alignment for {alignment_type}...")

    # Train-test split
    S_train, T_train, S_test, T_test = create_train_test_matrices(
        alignment_dict, entity_embeddings1, entity_embeddings2, test_size=0.1
    )

    if S_train is None or T_train is None:
        print(
            f"Skipping {alignment_type} alignment due to empty train-test matrices.")
        return

    # Apply Procrustes alignment (forward and reverse directions)
    print(f"Applying forward Procrustes alignment for {alignment_type}...")
    scaled_S_train, S_train_aligned, scaled_T_train, S_test_aligned, scaled_S_test, scaled_T_test, R = apply_procrustes_both_directions(
        S_train, T_train, S_test, T_test
    )

    print(f"Applying reverse Procrustes alignment for {alignment_type}...")
    scaled_T_train, T_train_aligned, scaled_S_train, T_test_aligned, scaled_T_test, scaled_S_test, R_reverse = apply_procrustes_both_directions(
        S_train, T_train, S_test, T_test, reverse=True
    )
    # Handle non-aligned embeddings
    print(f"Extracting non-aligned embeddings for {alignment_type}...")
    S_normalized, S_full_normalized, T_normalized, Sr_normalized, Tr_normalized,T_normalized_indices= extract_non_aligned_embeddings(
        alignment_dict=alignment_dict,
        embeddings1=entity_embeddings1,
        embeddings2=entity_embeddings2,
        S_train=scaled_S_train,
        T_train=scaled_T_train,
        S_test=scaled_S_test,
        T_test=scaled_T_test,
        relation_embeddings1=relation_embeddings1,
        relation_embeddings2=relation_embeddings2, output_folder=new_model_folder
    )
    
    print(f"Extracting non-aligned embeddings for {alignment_type}...")
    S_normalized, S_full_normalized, T_normalized, Sr_normalized, Tr_normalized,T_normalized_indices= extract_non_aligned_embeddings(
        alignment_dict=alignment_dict,
        embeddings1=entity_embeddings1,
        embeddings2=entity_embeddings2,
        S_train=scaled_S_train,
        T_train=scaled_T_train,
        S_test=scaled_S_test,
        T_test=scaled_T_test,
        relation_embeddings1=relation_embeddings1,
        relation_embeddings2=relation_embeddings2, output_folder=previous_model_folder
    )


    # Merge embeddings
    print(
        f"Merging aligned and non-aligned embeddings for {alignment_type}...")
    S_combined, S_combined_full, T_combined, averaged_embeddings_full = merge_embeddings(
        S_normalized, S_full_normalized, T_normalized, R, R_reverse,
        S_train_aligned, S_test_aligned,
        scaled_S_train, scaled_S_test,
        T_train_aligned, T_test_aligned,
        scaled_T_train, scaled_T_test, alignment_dict
    )

    # Create a DataFrame with the correct indices
    S_combined_df = pd.DataFrame(
        S_combined_full, index=entity_embeddings1.index)

    # Invert the mapping to index → URI for direct embedding access
    # index_to_uri = {v: k for k, v in revised_mapping.items()}

    # Check if the lengths of T_combined and T_normalized_indices match
    if len(T_combined) != len(T_normalized_indices):
        raise ValueError(f"Mismatch in lengths: T_combined has {len(T_combined)} rows, "
                        f"but T_normalized_indices has {len(T_normalized_indices)} elements.")

    # Create the DataFrame with T_combined as data and T_normalized_indices as the index
    T_combined_df = pd.DataFrame(T_combined, index=T_normalized_indices)

    # Combine S_combined and T_combined into a single DataFrame
    combined_matrix_final = pd.concat([S_combined_df, T_combined_df])

     # Step 1: Sort combined_matrix_final by index (URIs)
    sorted_combined_matrix = combined_matrix_final.sort_index()

    # Step 2: Extract sorted URIs
    sorted_uris = sorted_combined_matrix.index.tolist()
    # Step 3: Create entity-to-index mapping
    entity_to_idx = {uri: idx for idx, uri in enumerate(sorted_uris)}

    # Step 4: Save the entity-to-index mapping as a .p file
    for subfolder in ["new_model", "previous_model"]:
        output_path = os.path.join(output_folder, subfolder, "entity_to_idx.p")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            with open(output_path, "wb") as file:
                pickle.dump(entity_to_idx, file)
        except Exception as e:
            print(f"An error occurred while saving the .p file to {output_path}: {e}")


    # Step 5 (Optional): Save the sorted combined matrix (if needed)
    sorted_matrix_path = os.path.join(output_folder, "sorted_combined_matrix.npy")
    final_embeddings=sorted_combined_matrix.to_numpy()

    all_relation_embeddings_df,all_relation_embeddings_array = transform_merge_relation_embeddings(
        Sr_normalized, Tr_normalized, R, R_reverse, relation_embeddings1, relation_embeddings2
    )


    # Assume that the DataFrame's index is what you consider to be the relations
    relations = all_relation_embeddings_df.index

        # Create a mapping from each index (relation) to a unique numeric index
    relation_to_idx = {relation: idx for idx, relation in enumerate(relations)}
    
    try:
        # Initialize model
        model = initialize_models_and_update_embeddings(final_embeddings, all_relation_embeddings_array)

        # Extract URIs (index) from the DataFrame
        if 'all_relation_embeddings_df' not in locals():
            raise ValueError("all_relation_embeddings_df is not defined. Ensure it contains the required data.")
        uris = all_relation_embeddings_df.index.tolist()

        # Generate the mapping to indices
        relation_to_idx = {uri: idx for idx, uri in enumerate(uris)}

        for subfolder in ["new_model", "previous_model"]:
            sub_output_path = os.path.join(output_folder, subfolder)
            os.makedirs(sub_output_path, exist_ok=True)

            # Save full model
            model_path = os.path.join(sub_output_path, "model.pt")
            torch.save(model.state_dict(), model_path)
            print(f" Saved model to {model_path}")

            # Save relation_to_idx mapping
            relation_path = os.path.join(sub_output_path, "relation_to_idx.p")
            with open(relation_path, "wb") as f:
                pickle.dump(relation_to_idx, f)
            print(f" Saved relation-to-idx mapping to {relation_path}")

    except Exception as e:
        print(f" An error occurred during model/relation_to_idx saving: {e}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Process datasets for embedding extraction."
    )
    parser.add_argument("--embedding_folder", required=True,
                        help="Path to the embedding folder containing dataset subfolders.")
    parser.add_argument("--alignment_dict_path", required=True,
                        help="Path to pre-aligned alignment dictionary file.")
    parser.add_argument("--alignmentlimes_dict_path", required=True,
                        help="Path to LIMES alignment dictionary file.")
    parser.add_argument("--output_folder", required=True,
                        help="Folder to save processed outputs.")
    args = parser.parse_args()

    # Find dataset paths dynamically
    dataset1_path, dataset2_path = find_dataset_folders(args.embedding_folder)
    print(f"Dataset 1 Path: {dataset1_path}")
    print(f"Dataset 2 Path: {dataset2_path}")

    # Define output folder
    output_folder = os.path.join(
        args.output_folder, f"{os.path.basename(dataset1_path)}_{os.path.basename(dataset2_path)}"
    )
    os.makedirs(output_folder, exist_ok=True)

    # Load embeddings and related files
    dataset1_model, dataset1_entity_to_id, dataset1_relation_to_id = extract_files_from_directory(
        dataset1_path)
    dataset2_model, dataset2_entity_to_id, dataset2_relation_to_id = extract_files_from_directory(
        dataset2_path)

    print("Loading embeddings for both datasets...")
    entity_embeddings1, relation_embeddings1 = load_embeddings(
        dataset1_model, dataset1_entity_to_id, dataset1_relation_to_id)
    entity_embeddings2, relation_embeddings2 = load_embeddings(
        dataset2_model, dataset2_entity_to_id, dataset2_relation_to_id)


    # Normalize embeddings index
    entity_embeddings1 = remove_brackets_from_indices(entity_embeddings1)
    entity_embeddings2 = remove_brackets_from_indices(entity_embeddings2)
    
    entity_to_idx_path = os.path.join(dataset1_path, "entity_to_idx.p")
    if not os.path.exists(entity_to_idx_path):
        entity_to_idx_path = os.path.join(dataset1_path, "entity_to_idx.csv")

    relation_to_idx_path = os.path.join(dataset1_path, "relation_to_idx.p")
    if not os.path.exists(relation_to_idx_path):
        relation_to_idx_path = os.path.join(dataset1_path, "relation_to_idx.csv")

    train_set_path = os.path.join(dataset1_path, "train_set.npy")

    
    output_folder = os.path.join(
        args.output_folder, f"{os.path.basename(dataset1_path)}_{os.path.basename(dataset2_path)}"
        )

    new_model_folder = Path(output_folder) / "new_model"
    previous_model_folder = Path(output_folder) / "previous_model"

    # Load alignment dictionary
    pre_aligned_dict = build_alignment_dict(args.alignment_dict_path)

    cleaned_pre_aligned_dict = clean_dict(pre_aligned_dict)
    print(
        f"Processing pre-aligned alignment dictionary with {len(cleaned_pre_aligned_dict)} entries...")
    
    
    def load_entity_to_idx(file_path):
        """
        Load entity-to-index mapping from either a CSV or a Pickle file.

        - If it's a Pickle file (.p or .pkl), load it using pickle.
        - If it's a CSV file (.csv), assume it has two columns (entity, index) and load it into a dictionary.
        """
        file_extension = os.path.splitext(file_path)[-1].lower()

        if file_extension in ['.p', '.pkl']:  # Pickle file
            with open(file_path, 'rb') as f:
                entity_to_idx = pickle.load(f)
                if not isinstance(entity_to_idx, dict):
                    raise ValueError(f"Expected a dictionary in {file_path}, got {type(entity_to_idx)}")
                return entity_to_idx

        elif file_extension == '.csv':  # CSV file
            df = pd.read_csv(file_path, header=None, dtype=str)  # Ensure all data is loaded as strings
            if df.shape[1] != 2:
                raise ValueError(f"Expected CSV with 2 columns (entity, index), but found {df.shape[1]} columns in {file_path}")

            # Drop rows with NaN values
            df = df.dropna()

            # Convert to dictionary and strip unwanted characters
            entity_to_idx = {str(k).strip(): str(v).strip() for k, v in zip(df.iloc[:, 0], df.iloc[:, 1])}

            return entity_to_idx

        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Use .csv or .pkl/.p")
        

    d1_entity_to_idx = load_entity_to_idx(dataset1_entity_to_id)
    d2_entity_to_idx = load_entity_to_idx(dataset2_entity_to_id)

    # Clean entity_to_idx
    dataset1_entity_to_id1 = clean_dict(d1_entity_to_idx)
    dataset2_entity_to_id2 = clean_dict(d2_entity_to_idx)

    d1_relation_to_idx = load_entity_to_idx(dataset1_relation_to_id)
    d2_relation_to_idx = load_entity_to_idx(dataset2_relation_to_id)

    # Clean relation_to_idx
    dataset1_relation_to_id = clean_dict(d1_relation_to_idx)

    embeddings1_non_aligned, embeddings2_non_aligned = filter_and_print_non_aligned_embeddings(
        cleaned_pre_aligned_dict, entity_embeddings1, entity_embeddings2)
    
    
    new_model_folder = os.path.join(output_folder, "new_model")
    previous_model_folder = os.path.join(output_folder, "previous_model")
    os.makedirs(new_model_folder, exist_ok=True)
    os.makedirs(previous_model_folder, exist_ok=True)

    
    generate_train_set_with_uris(
        entity_to_idx_path=os.path.join(dataset1_path, "entity_to_idx.csv"),
        relation_to_idx_path=os.path.join(dataset1_path, "relation_to_idx.csv"),
        train_set_path=os.path.join(dataset1_path, "train_set.npy"),
        output_path=os.path.join(new_model_folder, "train_set_with_relation1.npy")
    )
    generate_train_set_with_uris(
        entity_to_idx_path=os.path.join(dataset1_path, "entity_to_idx.csv"),
        relation_to_idx_path=os.path.join(dataset1_path, "relation_to_idx.csv"),
        train_set_path=os.path.join(dataset1_path, "train_set.npy"),
        output_path=os.path.join(previous_model_folder, "train_set_with_relation1.npy")
    )
   
    generate_train_set_with_uris(
    entity_to_idx_path=os.path.join(dataset2_path, "entity_to_idx.csv"),
    relation_to_idx_path=os.path.join(dataset2_path, "relation_to_idx.csv"),
    train_set_path=os.path.join(dataset2_path, "train_set.npy"),
    output_path=os.path.join(new_model_folder, "train_set_with_relation2.npy")
    )
    generate_train_set_with_uris(
        entity_to_idx_path=os.path.join(dataset2_path, "entity_to_idx.csv"),
        relation_to_idx_path=os.path.join(dataset2_path, "relation_to_idx.csv"),
        train_set_path=os.path.join(dataset2_path, "train_set.npy"),
        output_path=os.path.join(previous_model_folder, "train_set_with_relation2.npy")
    )

    def concat_train_sets(train_set_path1, train_set_path2, output_path):
        """
        Concatenates two train.npy files into one.

        Args:
            train_set_path1 (str): Path to the first NumPy file containing triples with URIs.
            train_set_path2 (str): Path to the second NumPy file containing triples with URIs.
            output_path (str): Path to save the concatenated NumPy file.

        Returns:
            None
        """
        try:
            # Load the first train.npy file
            train_data1 = np.load(train_set_path1, allow_pickle=True)
            # Load the second train.npy file
            train_data2 = np.load(train_set_path2, allow_pickle=True)
        except Exception as e:
            print(f"Error loading train.npy files: {e}")
            return

        # Concatenate the two datasets
        concatenated_data = np.concatenate([train_data1, train_data2], axis=0)

        try:
            # Save the concatenated dataset
            np.save(output_path, concatenated_data)
            print(f"Concatenated train.npy saved to {output_path}")
        except Exception as e:
            print(f"Error saving concatenated train.npy: {e}")

    # New model path
    concat_train_sets(
        train_set_path1=os.path.join(new_model_folder, "train_set_with_relation1.npy"),
        train_set_path2=os.path.join(new_model_folder, "train_set_with_relation2.npy"),
        output_path=os.path.join(new_model_folder, "concatenated_train_set_with_relation.npy"))


    # Previous model path
    concat_train_sets(
        train_set_path1=os.path.join(new_model_folder, "train_set_with_relation1.npy"),
        train_set_path2=os.path.join(previous_model_folder, "train_set_with_relation2.npy"),
        output_path=os.path.join(previous_model_folder, "concatenated_train_set_with_relation.npy"))

        
    def replace_fr_with_en(train_set_path, alignment_dict, updated_train_set_path):
        """
        Replaces 'fr' DBpedia URIs with 'en' DBpedia URIs in the training dataset using a sameAs alignment dictionary.

        Args:
            train_set_path (str): Path to the NumPy file containing triples with URIs.
            alignment_dict (dict): A dictionary where keys are 'fr' DBpedia URIs and values are 'en' DBpedia URIs.
            updated_train_set_path (str): Path to save the updated NumPy file.

        Returns:
            None
        """
        # Step 2: Invert the alignment dictionary (EN → FR)
        inverted_alignment_dict = {v: k for k, v in alignment_dict.items()}

        try:
            train_set_with_uris = np.load(train_set_path, allow_pickle=True)
        except Exception as e:
            print(f"\n Error loading train.npy: {e}")
            return

        # Normalize function (removes spaces & lowercase)
        def normalize_uri(uri):
            return uri.strip().lower()

        # Step 6: Replace 'fr' URIs with 'en' URIs
        updated_train_set_with_uris = []
        replaced_count = 0
        unchanged_count = 0

        for triple in train_set_with_uris:
            subject, predicate, obj = map(normalize_uri, triple)  # Normalize URIs
            original_subject, original_object = subject, obj

            # Replace using the inverted alignment dictionary
            subject = inverted_alignment_dict.get(subject, subject)
            obj = inverted_alignment_dict.get(obj, obj)

            # Debug - Print replacements made
            if original_subject != subject or original_object != obj:
                replaced_count += 1
            else:
                unchanged_count += 1

            updated_train_set_with_uris.append([subject, predicate, obj])

        # Step 8: Save the updated training dataset
        try:
            np.save(updated_train_set_path, np.array(updated_train_set_with_uris, dtype=object))
        except Exception as e:
            print(f"\n Error saving updated training dataset: {e}")

    # File pathstrain_set_path = os.path.join(new_model_folder, "concatenated_train_set_with_relation.npy")
    train_set_path = os.path.join(new_model_folder, "concatenated_train_set_with_relation.npy")

    for model_path in [new_model_folder, previous_model_folder]:
        replace_fr_with_en(
            train_set_path=os.path.join(model_path, "concatenated_train_set_with_relation.npy"),
            alignment_dict=cleaned_pre_aligned_dict,
            updated_train_set_path=os.path.join(model_path, "updated_concatenated_train_set_with_relation.npy")
        )

    def revise_train_set_with_indexes(
        train_with_uris_path, entity_to_idx_path, relation_to_idx_path, output_path
    ):
        """
        Revises triples in a train.npy file from URIs back to numerical indices using entity-to-index and relation-to-index mappings.

        Args:
            train_with_uris_path (str): Path to the NumPy file containing triples with URIs.
            entity_to_idx_path (str): Path to the pickle file containing entity-to-index mapping.
            relation_to_idx_path (str): Path to the pickle file containing relation-to-index mapping.
            output_path (str): Path to save the output NumPy file with triples represented by indices.

        Returns:
            None
        """
        try:
            # Load the entity-to-index mapping
            with open(entity_to_idx_path, "rb") as file:
                entity_to_idx = pickle.load(file)
        except Exception as e:
            print(f"Error loading entity_to_idx mapping: {e}")
            return

        try:
            # Load the relation-to-index mapping
            with open(relation_to_idx_path, "rb") as file:
                relation_to_idx = pickle.load(file)
        except Exception as e:
            print(f"Error loading relation_to_idx mapping: {e}")
            return

        try:
            # Load the train.npy file with URIs
            train_with_uris = np.load(train_with_uris_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading train_with_uris.npy: {e}")
            return

        # Replace URIs with indices
        train_with_indices = []
        for triple in train_with_uris:
            subject = entity_to_idx.get(triple[0], f"Unknown Index ({triple[0]})")
            predicate = relation_to_idx.get(triple[1], f"Unknown Index ({triple[1]})")
            obj = entity_to_idx.get(triple[2], f"Unknown Index ({triple[2]})")
            train_with_indices.append([subject, predicate, obj])

        try:
            # Save the revised dataset with indices
            np.save(output_path, np.array(train_with_indices, dtype=object))
        except Exception as e:
            print(f"Error saving revised train.npy: {e}")


    for model_path in [new_model_folder, previous_model_folder]:
        revise_train_set_with_indexes(
            train_with_uris_path=os.path.join(model_path, "updated_concatenated_train_set_with_relation.npy"),
            entity_to_idx_path=os.path.join(model_path, "entity_to_idx.p"),
            relation_to_idx_path=os.path.join(model_path, "relation_to_idx.p"),
            output_path=os.path.join(model_path, "train_set.npy")
        )


    if cleaned_pre_aligned_dict:
        print(
            f"Starting alignment pipeline with {len(cleaned_pre_aligned_dict)} entries.")
        procrustes_alignment_pipeline(
            cleaned_pre_aligned_dict, entity_embeddings1, entity_embeddings2,
            relation_embeddings1, relation_embeddings2, output_folder,
            "pre_aligned", cleaned_pre_aligned_dict,
            new_model_folder=new_model_folder,
            previous_model_folder=previous_model_folder
            )

    else:
        print("No valid pre-aligned alignment dictionary found. Skipping pre-aligned processing.")

    # Handle LIMES alignment dictionary
    limes_dict = build_alignment_dict(args.alignmentlimes_dict_path)
    if limes_dict:
        cleaned_limes_dict = clean_dict(limes_dict)
        print(
            f"Processing LIMES alignment dictionary with {len(cleaned_limes_dict)} entries...")
        procrustes_alignment_pipeline(
            cleaned_limes_dict, entity_embeddings1, entity_embeddings2,
            relation_embeddings1, relation_embeddings2, output_folder, "limes", cleaned_pre_aligned_dict
        )
    else:
        print("No valid LIMES alignment dictionary found. Skipping LIMES processing.")

    print("Processing completed successfully.")
    
    # After processing
    for temp_file in [
        os.path.join(new_model_folder, "train_set_with_relation1.npy"),
        os.path.join(new_model_folder, "train_set_with_relation2.npy"),
        os.path.join(new_model_folder, "concatenated_train_set_with_relation.npy"),
        os.path.join(new_model_folder, "updated_concatenated_train_set_with_relation.npy"),
        os.path.join(previous_model_folder, "train_set_with_relation1.npy"),
        os.path.join(previous_model_folder, "train_set_with_relation2.npy"),
        os.path.join(previous_model_folder, "concatenated_train_set_with_relation.npy"),
        os.path.join(previous_model_folder, "updated_concatenated_train_set_with_relation.npy")
    ]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            #print(f"Deleted temporary file: {temp_file}")
            

    def run_link_prediction_evaluation(model_folder):
        model_path = os.path.join(model_folder, "model.pt")
        test_path = os.path.join(model_folder, "test_triples.txt")

        if not os.path.exists(model_path) or not os.path.exists(test_path):
            print(f" Skipping evaluation — missing files in: {model_folder}")
            return

        print(f"\n Running link prediction evaluation for {model_folder}...")

        # Load model
        model = KGE(path=model_folder)

        # Load test triples
        all_triples = pd.read_csv(
            test_path, sep="\s+", header=None, names=["subject", "relation", "object"], dtype=str
        ).values.tolist()

        # Build vocab
        er_vocab = get_er_vocab(all_triples)
        re_vocab = get_re_vocab(all_triples)

        # Evaluate
        results = evaluate_link_prediction_performance(
            model=model, triples=all_triples, er_vocab=er_vocab, re_vocab=re_vocab
        )

        print(f" Results for {model_folder}:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")

        # Save results to file
        with open(os.path.join(model_folder, "link_prediction_results.txt"), "w") as f:
            for k, v in results.items():
                f.write(f"{k}: {v:.4f}\n")

    # Call it for new_model only (as you requested earlier)
    run_link_prediction_evaluation(new_model_folder)



if __name__ == "__main__":
    main()

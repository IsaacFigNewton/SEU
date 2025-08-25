#!/usr/bin/env python3
"""
Simple example of using the SEU package for entity alignment.

This script demonstrates the core workflow from main.ipynb using
only the essential functions.
"""

import os
import numpy as np
import tensorflow as tf
import seu

# Set up environment
seed = 12345
np.random.seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU only

def main():
    print("SEU Entity Alignment Example")
    print("=" * 40)
    
    # Step 1: Download and load embeddings
    print("Step 1: Loading word embeddings...")
    try:
        # Try to load existing embeddings first
        if os.path.exists("glove.6B.300d.txt"):
            word_vecs = seu.load_word_vectors("glove.6B.300d.txt")
            print(f"Loaded existing embeddings: {len(word_vecs)} words")
        else:
            print("Downloading GloVe embeddings...")
            embedding_path = seu.download_and_extract_file(
                "http://nlp.stanford.edu/data/glove.6B.zip",
                "glove.6B.300d.txt",
                "./glove.6B.300d.txt"
            )
            word_vecs = seu.load_word_vectors(embedding_path)
            print(f"Downloaded and loaded embeddings: {len(word_vecs)} words")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        print("Please ensure internet connection or download embeddings manually")
        return
    
    # Step 2: Load dataset
    print("\nStep 2: Loading dataset...")
    try:
        # Load knowledge graph data
        dataset_path = "KGs/dbp_ja_en/"
        entities_path = "translated_ent_name/dbp_ja_en.json"
        
        if not os.path.exists(dataset_path):
            print(f"Dataset not found at {dataset_path}")
            print("Please ensure the KG dataset is available")
            return
            
        if not os.path.exists(entities_path):
            print(f"Entity names not found at {entities_path}")
            print("Please ensure the entity names file is available")
            return
        
        all_triples, node_size, rel_size = seu.load_triples(dataset_path, reverse=True)
        train_pair, test_pair = seu.load_aligned_pair(dataset_path, ratio=0.0)  # Use all for testing
        ent_names = seu.load_entity_names(entities_path)
        
        print(f"Loaded {len(all_triples)} triples, {node_size} nodes, {rel_size} relations")
        print(f"Test pairs: {len(test_pair)}")
        print(f"Entity names: {len(ent_names)}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure dataset files are available")
        return
    
    # Step 3: Generate features
    print("\nStep 3: Generating features...")
    feature = seu.generate_features(ent_names, word_vecs, node_size, mode="hybrid-level")
    print(f"Generated features with shape: {feature.shape}")
    
    # Step 4: Build graph and calculate similarities
    print("\nStep 4: Building graph and calculating similarities...")
    sparse_rel_matrix = seu.build_sparse_adjacency_matrix(all_triples, node_size)
    sims = seu.calculate_similarities(test_pair, feature, sparse_rel_matrix, depth=2)
    print(f"Calculated similarity matrix: {sims.shape}")
    
    # Step 5: Run solvers
    print("\nStep 5: Running alignment solvers...")
    
    # Hungarian algorithm
    print("\nRunning Hungarian algorithm...")
    hungarian_result = seu.hungarian_solve(sims)
    print("Hungarian results:")
    seu.test(hungarian_result, "hungarian")
    
    # Sinkhorn algorithm
    print("\nRunning Sinkhorn algorithm...")
    sinkhorn_result = seu.sinkhorn_solve(sims, temperature=50.0, max_iterations=10)
    print("Sinkhorn results:")
    seu.test(sinkhorn_result, "sinkhorn")
    
    print("\nEntity alignment completed successfully!")


if __name__ == "__main__":
    main()
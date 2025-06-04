#!/usr/bin/env python3
"""
Script to create CodeSearchNet dataset for CodeBERT fine-tuning.
Creates training and validation splits with balanced positive and negative examples.
"""

import os
import random
import argparse
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = "../data/codesearch/train_valid"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create CodeSearchNet dataset for CodeBERT"
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["go", "java", "javascript", "php", "python", "ruby"],
        help="Programming language to create dataset for",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/codesearch/train_valid",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=50000,
        help="Number of training examples (positive + negative)",
    )
    parser.add_argument(
        "--valid_size",
        type=int,
        default=10000,
        help="Number of validation examples (positive + negative)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def load_and_process_dataset(language):
    """Load and process the CodeSearchNet dataset for the specified language."""
    print(f"Loading {language} dataset...")
    dataset = load_dataset("code_search_net", language)

    # Process the dataset to extract (docstring, code) pairs
    processed_data = []
    for split in ["train", "validation"]:
        for example in tqdm(dataset[split], desc=f"Processing {split} split"):
            # Skip examples with empty docstring or code
            if (
                example["func_documentation_string"].strip()
                and example["func_code_string"].strip()
            ):
                processed_data.append(
                    {
                        "docstring": example["func_documentation_string"]
                        .replace("\n", " ")
                        .strip(),
                        "code": example["func_code_string"].replace("\n", " ").strip(),
                    }
                )

    return processed_data


def create_negative_pairs(data, num_pairs):
    """Create negative pairs by randomly pairing docstrings with non-matching code."""
    negative_pairs = []
    data_size = len(data)

    # Ensure we don't create more pairs than possible combinations
    num_pairs = min(num_pairs, data_size * (data_size - 1) // 2)

    # Create a set to track used pairs to avoid duplicates
    used_pairs = set()

    while len(negative_pairs) < num_pairs:
        i, j = random.sample(range(data_size), 2)

        # Ensure we don't create a positive pair
        if i != j and (i, j) not in used_pairs and (j, i) not in used_pairs:
            negative_pairs.append((data[i]["docstring"], data[j]["code"]))
            used_pairs.add((i, j))

    return negative_pairs


def write_dataset(data, output_file, num_samples, negative_ratio=1.0):
    """Write dataset to file in the required format."""
    # Calculate number of positive and negative samples
    num_positive = min(num_samples // 2, len(data))
    num_negative = min(int(num_positive * negative_ratio), num_samples - num_positive)

    # Select positive examples
    positive_samples = data[:num_positive]

    # Create negative examples
    negative_samples = create_negative_pairs(data, num_negative)

    # Write to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        # Write positive examples
        for example in positive_samples:
            f.write(
                f"1<CODESPLIT><CODESPLIT><CODESPLIT>{example['docstring']}<CODESPLIT>{example['code']}\n"
            )

        # Write negative examples
        for docstring, code in negative_samples:
            f.write(f"0<CODESPLIT><CODESPLIT><CODESPLIT>{docstring}<CODESPLIT>{code}\n")

    print(
        f"Created {output_file} with {num_positive} positive and {len(negative_samples)} negative examples"
    )


def main():
    args = parse_args()
    random.seed(args.seed)

    # Load and process the dataset
    data = load_and_process_dataset(args.language)
    random.shuffle(data)  # Shuffle the data

    # Split into train and validation
    total_size = len(data)
    train_size = min(
        args.train_size // 2, total_size // 2
    )  # Half for positive, half for negative
    valid_size = min(args.valid_size // 2, (total_size - train_size) // 2)

    train_data = data[:train_size]
    valid_data = data[train_size : train_size + valid_size]

    # Create output directory structure
    output_dir = os.path.join(DATA_DIR, args.language)
    os.makedirs(output_dir, exist_ok=True)

    # Create training set
    train_file = os.path.join(output_dir, "train.txt")
    write_dataset(train_data, train_file, args.train_size)

    # Create validation set
    valid_file = os.path.join(output_dir, "valid.txt")
    write_dataset(valid_data, valid_file, args.valid_size)

    print(f"\nDataset creation complete for {args.language}.")
    print(f"Training set: {os.path.abspath(train_file)}")
    print(f"Validation set: {os.path.abspath(valid_file)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script to verify the knowledge distillation setup
"""

import os
import sys

# Add project root to path
sys.path.append("/Users/omkarkabde/Desktop/KD")


def test_directory_structure(test_dataset="test_dataset"):
    """Test if required directories exist or can be created"""
    print("\nTesting directory structure...")

    required_dirs = [
        "checkpoints",
        f"checkpoints/{test_dataset}",
        f"checkpoints/{test_dataset}/teachers",
        f"checkpoints/{test_dataset}/AT",
        f"checkpoints/{test_dataset}/BYOT",
        f"checkpoints/{test_dataset}/DML",
        f"checkpoints/{test_dataset}/FitNets",
        "logs",
        f"logs/{test_dataset}",
        f"logs/{test_dataset}/teachers",
        f"logs/{test_dataset}/AT",
        f"logs/{test_dataset}/BYOT",
        f"logs/{test_dataset}/DML",
        f"logs/{test_dataset}/FitNets",
    ]

    for dir_path in required_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)

        except Exception as e:
            print(f"❌ Directory {dir_path} creation failed: {e}")
            return False
    print("✅ All directories ready")

    return True

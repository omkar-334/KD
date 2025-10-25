#!/usr/bin/env python3
"""
Test script to verify the fixes work
"""

import sys

import torch
import torch.nn as nn

# Add project root to path
sys.path.append("/Users/omkarkabde/Desktop/KD")


def test_model_outputs():
    """Test that models return correct output shapes"""
    print("Testing model output shapes...")

    from shared_utils import get_model

    # Test ResNet18
    model = get_model({"model": "ResNet18"}, num_classes=4)
    model.eval()

    # Create dummy input
    x = torch.randn(2, 3, 224, 224)  # batch_size=2, channels=3, height=224, width=224
    y = torch.randint(0, 4, (2,))  # batch_size=2, num_classes=4

    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")

    # Test forward pass
    with torch.no_grad():
        output = model(x)
        print(f"Model output type: {type(output)}")
        print(f"Model output shape: {output.shape}")

        # Test loss computation
        criterion = nn.CrossEntropyLoss()
        try:
            loss = criterion(output, y)
            print(f"✅ Loss computation successful: {loss.item():.4f}")
        except Exception as e:
            print(f"❌ Loss computation failed: {e}")
            return False

    return True


def test_at_loss():
    """Test AT loss computation"""
    print("\nTesting AT loss computation...")

    from AT.main import compute_at_loss
    from shared_utils import get_model

    # Create models
    student_model = get_model({"model": "ResNet18"}, num_classes=4)
    teacher_model = get_model({"model": "ResNet50"}, num_classes=4)

    student_model.eval()
    teacher_model.eval()

    # Create dummy input
    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 4, (2,))

    config = {"model": "ResNet18", "alpha": 0.1, "beta": 1e3, "temperature": 4.0}

    try:
        loss, loss_dict = compute_at_loss(student_model, teacher_model, x, y, config)
        print(f"✅ AT loss computation successful: {loss.item():.4f}")
        print(f"Loss dict: {loss_dict}")
        return True
    except Exception as e:
        print(f"❌ AT loss computation failed: {e}")
        return False


def test_byot_loss():
    """Test BYOT loss computation"""
    print("\nTesting BYOT loss computation...")

    from BYOT.main import compute_byot_loss
    from shared_utils import get_model

    # Create models
    student_model = get_model({"model": "ResNet18"}, num_classes=4)
    teacher_model = get_model({"model": "ResNet50"}, num_classes=4)

    student_model.eval()
    teacher_model.eval()

    # Create dummy input
    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 4, (2,))

    config = {"model": "ResNet18", "alpha": 0.1, "beta": 0.1, "temperature": 4.0}

    try:
        loss, loss_dict = compute_byot_loss(student_model, x, y, config, teacher_model)
        print(f"✅ BYOT loss computation successful: {loss.item():.4f}")
        print(f"Loss dict: {loss_dict}")
        return True
    except Exception as e:
        print(f"❌ BYOT loss computation failed: {e}")
        return False


def test_fitnets_loss():
    """Test FitNets loss computation"""
    print("\nTesting FitNets loss computation...")

    from FitNets.main import compute_fitnets_loss
    from shared_utils import get_model

    # Create models
    student_model = get_model({"model": "ResNet18"}, num_classes=4)
    teacher_model = get_model({"model": "ResNet50"}, num_classes=4)

    student_model.eval()
    teacher_model.eval()

    # Create dummy input
    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 4, (2,))

    config = {"model": "ResNet18", "alpha": 0.1, "beta": 1e3, "temperature": 4.0}

    try:
        loss, loss_dict = compute_fitnets_loss(
            student_model, teacher_model, x, y, config
        )
        print(f"✅ FitNets loss computation successful: {loss.item():.4f}")
        print(f"Loss dict: {loss_dict}")
        return True
    except Exception as e:
        print(f"❌ FitNets loss computation failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Testing Knowledge Distillation Fixes")
    print("=" * 50)

    tests = [
        ("Model Output Shapes", test_model_outputs),
        ("AT Loss Computation", test_at_loss),
        ("BYOT Loss Computation", test_byot_loss),
        ("FitNets Loss Computation", test_fitnets_loss),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Fixes are working.")
    else:
        print("❌ Some tests failed. Please check the issues above.")

    return passed == total


if __name__ == "__main__":
    main()

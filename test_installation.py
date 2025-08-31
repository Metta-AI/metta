#!/usr/bin/env python3
"""
Simple test script to verify Metta AI installation is working
"""

def test_basic_imports():
    """Test that basic packages can be imported"""
    print("Testing basic package imports...")

    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False

    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False

    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")

        # Test CUDA if available
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CUDA not available (CPU-only mode)")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False

    return True

def test_metta_imports():
    """Test that Metta-specific packages can be imported"""
    print("\nTesting Metta-specific imports...")

    # Test metta common
    try:
        from metta.common.util.fs import get_repo_root
        print("✅ Metta common imported successfully")
    except ImportError as e:
        print(f"❌ Metta common import failed: {e}")
        return False

    # Test metta agent
    try:
        import metta.agent
        print("✅ Metta agent imported successfully")
    except ImportError as e:
        print(f"❌ Metta agent import failed: {e}")
        return False

    # Test metta app backend
    try:
        import metta.app_backend
        print("✅ Metta app backend imported successfully")
    except ImportError as e:
        print(f"❌ Metta app backend import failed: {e}")
        return False

    return True

def test_tensor_operations():
    """Test basic tensor operations"""
    print("\nTesting tensor operations...")

    try:
        import torch
        import numpy as np

        # Create a simple tensor
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x + y

        print(f"✅ Tensor addition works: {z.shape}")
        print(f"   Result: {z}")

        # Test numpy integration
        np_array = np.random.randn(3, 3)
        torch_tensor = torch.from_numpy(np_array)
        result = torch_tensor * 2

        print(f"✅ NumPy-Torch integration works: {result.shape}")

        return True
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing Metta AI Installation")
    print("=" * 50)

    success = True

    success &= test_basic_imports()
    success &= test_metta_imports()
    success &= test_tensor_operations()

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Metta AI installation is working correctly.")
        print("\nNext steps:")
        print("1. Try running: uv run jupyter notebook")
        print("2. Open experiments/notebooks/01-hello-world.ipynb")
        print("3. For full game demos, you'll need to install Visual Studio Build Tools")
        print("   and uncomment metta-mettagrid in pyproject.toml")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("You may need to reinstall some packages or fix dependencies.")

    return success

if __name__ == "__main__":
    main()

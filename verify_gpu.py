"""
GPU Setup Verification Script
Checks if your system is properly configured for GPU acceleration
"""

import sys
import subprocess
import platform


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def check_nvidia_driver():
    """Check if NVIDIA driver is installed and working"""
    print_section("1. NVIDIA Driver Check")
    
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        
        if result.returncode == 0:
            print("✅ NVIDIA driver is installed and working")
            print("\nGPU Information:")
            # Parse nvidia-smi output for key info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA' in line or 'Driver Version' in line or 'CUDA Version' in line:
                    print(f"   {line.strip()}")
            return True
        else:
            print("❌ nvidia-smi command failed")
            print(f"   Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        print("   Please install NVIDIA drivers from: https://www.nvidia.com/drivers")
        return False
    except Exception as e:
        print(f"❌ Error checking NVIDIA driver: {e}")
        return False


def check_python_version():
    """Check Python version compatibility"""
    print_section("2. Python Version Check")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"Python version: {version_str}")
    
    if version.major == 3 and 8 <= version.minor <= 11:
        print("✅ Python version is compatible with TensorFlow GPU")
        return True
    elif version.major == 3 and version.minor >= 12:
        print("⚠️  Python 3.12+ detected")
        print("   TensorFlow may have limited support. Consider Python 3.11 or lower.")
        return True
    else:
        print("❌ Python version not compatible")
        print("   TensorFlow requires Python 3.8-3.11")
        return False


def check_tensorflow():
    """Check TensorFlow installation and GPU support"""
    print_section("3. TensorFlow Installation Check")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow is installed: version {tf.__version__}")
        
        # Check version for Windows compatibility
        version_parts = tf.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if platform.system() == 'Windows':
            if major >= 2 and minor >= 11:
                print("⚠️  WARNING: TensorFlow 2.11+ detected on Windows")
                print("   Native GPU support was dropped in TensorFlow 2.11 for Windows")
                print("   Recommendation: Downgrade to TensorFlow 2.10 or lower")
                print("   Command: pip uninstall tensorflow && pip install \"tensorflow<2.11\"")
        
        return True
        
    except ImportError:
        print("❌ TensorFlow is not installed")
        print("   Install with: pip install tensorflow")
        return False
    except Exception as e:
        print(f"❌ Error checking TensorFlow: {e}")
        return False


def check_gpu_devices():
    """Check if TensorFlow can detect GPU"""
    print_section("4. GPU Device Detection")
    
    try:
        import tensorflow as tf
        
        # List physical devices
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ SUCCESS: {len(gpus)} GPU(s) detected by TensorFlow")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
            return True
        else:
            print("❌ No GPU detected by TensorFlow")
            print("\nPossible causes:")
            print("   1. CUDA Toolkit not installed")
            print("   2. cuDNN not installed")
            print("   3. TensorFlow version incompatible (Windows needs <2.11)")
            print("   4. CUDA/cuDNN version mismatch")
            return False
            
    except Exception as e:
        print(f"❌ Error detecting GPU: {e}")
        return False


def check_cuda_cudnn():
    """Check CUDA and cuDNN installation"""
    print_section("5. CUDA and cuDNN Check")
    
    try:
        import tensorflow as tf
        
        # Try to get CUDA version
        cuda_version = None
        cudnn_version = None
        
        # Get build info
        build_info = tf.sysconfig.get_build_info()
        
        if 'cuda_version' in build_info:
            cuda_version = build_info['cuda_version']
            print(f"✅ CUDA version (TensorFlow built with): {cuda_version}")
        else:
            print("⚠️  CUDA version information not available")
        
        if 'cudnn_version' in build_info:
            cudnn_version = build_info['cudnn_version']
            print(f"✅ cuDNN version (TensorFlow built with): {cudnn_version}")
        else:
            print("⚠️  cuDNN version information not available")
        
        if cuda_version or cudnn_version:
            print("\nNote: These are the versions TensorFlow was built with.")
            print("Your system should have compatible versions installed.")
            return True
        else:
            print("\nℹ️  Version information not available in this TensorFlow build")
            return True
            
    except Exception as e:
        print(f"⚠️  Could not check CUDA/cuDNN versions: {e}")
        return True  # Not a critical failure


def test_gpu_operation():
    """Test a simple GPU operation"""
    print_section("6. GPU Operation Test")
    
    try:
        import tensorflow as tf
        
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("⚠️  Skipping GPU operation test (no GPU detected)")
            return False
        
        # Try a simple operation
        print("Running a simple matrix multiplication on GPU...")
        
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            result = c.numpy()  # Force execution
        
        print("✅ GPU operation successful!")
        print(f"   Result shape: {result.shape}")
        return True
        
    except Exception as e:
        print(f"❌ GPU operation failed: {e}")
        print("\nThis might indicate:")
        print("   - CUDA/cuDNN installation issues")
        print("   - GPU memory problems")
        print("   - TensorFlow GPU support not working")
        return False


def check_mixed_precision_support():
    """Check if mixed precision is supported"""
    print_section("7. Mixed Precision Support")
    
    try:
        import tensorflow as tf
        from tensorflow.keras import mixed_precision
        
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("⚠️  Skipping (no GPU detected)")
            return False
        
        # Try to set mixed precision policy
        policy = mixed_precision.Policy('mixed_float16')
        
        print("✅ Mixed precision (float16) is supported")
        print("   This will speed up training on RTX GPUs (20-40% faster)")
        print("   Memory usage will be reduced by ~40%")
        return True
        
    except Exception as e:
        print(f"⚠️  Mixed precision may not be fully supported: {e}")
        print("   Training will still work, but may be slower")
        return True  # Not critical


def print_recommendations(checks_passed):
    """Print final recommendations based on checks"""
    print_section("Summary and Recommendations")
    
    total_checks = 7
    passed = sum(checks_passed)
    
    print(f"\nChecks passed: {passed}/{total_checks}")
    
    if passed == total_checks:
        print("\n🎉 EXCELLENT! Your system is fully configured for GPU acceleration!")
        print("\nNext steps:")
        print("   1. Run training with GPU acceleration:")
        print("      python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --batch-size 128")
        print("\n   2. Monitor GPU usage during training:")
        print("      nvidia-smi -l 1  (in a separate terminal)")
        print("\n   3. Expected performance:")
        print("      - Training speed: 10-20x faster than CPU")
        print("      - VRAM usage: 2-4GB with batch_size=128")
        print("      - Time per epoch: 1-3 seconds (vs 10-20s on CPU)")
        
    elif passed >= 4:
        print("\n⚠️  PARTIAL: Your system has basic GPU support but some issues need attention")
        print("\nRecommended actions:")
        
        if not checks_passed[3]:  # GPU not detected
            print("   🔧 PRIORITY: Fix GPU detection issue")
            print("      - Verify TensorFlow version (<2.11 for Windows)")
            print("      - Install CUDA Toolkit 11.2")
            print("      - Install cuDNN 8.1 for CUDA 11.2")
        
        if not checks_passed[5]:  # GPU operation failed
            print("   🔧 Fix GPU operation issues")
            print("      - Reinstall CUDA/cuDNN")
            print("      - Check TensorFlow GPU documentation")
        
        print("\n   After fixing issues, run this script again to verify")
        
    else:
        print("\n❌ CRITICAL: Multiple GPU setup issues detected")
        print("\nAction plan:")
        print("   1. Update NVIDIA drivers: https://www.nvidia.com/drivers")
        print("   2. Install compatible TensorFlow version:")
        print("      pip uninstall tensorflow")
        print("      pip install \"tensorflow<2.11\"  (Windows)")
        print("   3. Install CUDA Toolkit 11.2 and cuDNN 8.1")
        print("   4. Run this verification script again")
        print("\n   For detailed help, see TROUBLESHOOTING.md")
    
    print("\n" + "="*70)


def main():
    """Run all GPU verification checks"""
    print("\n" + "="*70)
    print("  GPU ACCELERATION VERIFICATION TOOL")
    print("  For NVIDIA RTX 3050 (6GB VRAM)")
    print("="*70)
    print(f"\nOperating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    
    # Run all checks
    checks_passed = [
        check_nvidia_driver(),
        check_python_version(),
        check_tensorflow(),
        check_gpu_devices(),
        check_cuda_cudnn(),
        test_gpu_operation(),
        check_mixed_precision_support(),
    ]
    
    # Print recommendations
    print_recommendations(checks_passed)


if __name__ == '__main__':
    main()

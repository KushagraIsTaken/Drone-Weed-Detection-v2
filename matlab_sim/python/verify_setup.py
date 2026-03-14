import os
import sys

def check_setup():
    print("=== UAV Simulation Setup Verification ===")
    
    # Check packages
    missing_packages = []
    try:
        import onnxruntime
    except ImportError:
        missing_packages.append("onnxruntime")
        
    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")
        
    try:
        import cv2
    except ImportError:
        missing_packages.append("opencv-python")
        
    try:
        import PIL
    except ImportError:
        missing_packages.append("Pillow")
        
    try:
        import tqdm
    except ImportError:
        missing_packages.append("tqdm")
        
    try:
        import json
    except ImportError:
        missing_packages.append("json (built-in)")
        
    if missing_packages:
        print("[FAIL] Missing Python packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        sys.exit(1)
    else:
        print("[PASS] All required Python packages are installed.")
        
    # Check models
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    expected_models = [
        "teacher.onnx",
        "small_baseline.onnx",
        "nano_baseline.onnx",
        "nano_feature_kd.onnx",
        "small_feature_kd.onnx",
        "nano_pseudo_kd.onnx",
        "small_pseudo_kd.onnx"
    ]
    
    import onnxruntime as ort
    
    all_models_present = True
    for model_name in expected_models:
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            print(f"[FAIL] Missing model file: {model_name} in {model_dir}")
            all_models_present = False
        elif os.path.getsize(model_path) < 1000:
            print(f"[FAIL] Model {model_name} is an LFS pointer (not downloaded). Size: {os.path.getsize(model_path)} bytes.")
            all_models_present = False
        else:
            try:
                # Try CoreML first on Mac, then CPU
                providers = []
                if 'CoreMLExecutionProvider' in ort.get_available_providers():
                    providers.append('CoreMLExecutionProvider')
                providers.append('CPUExecutionProvider')
                
                session = ort.InferenceSession(model_path, providers=providers)
                input_details = session.get_inputs()[0]
                output_details = session.get_outputs()[0]
                print(f"[PASS] Model {model_name} loaded successfully ({session.get_providers()[0]}). Input: {input_details.shape}, Output: {output_details.shape}")
            except Exception as e:
                print(f"[FAIL] Model {model_name} failed to load: {e}")
                all_models_present = False
                
    if not all_models_present:
        print("Please place the 6 required ONNX model files in the matlab_sim/models/ directory.")
        
    # Check data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    images_dir = os.path.join(data_dir, "test_images")
    labels_dir = os.path.join(data_dir, "test_labels")
    
    if not os.path.exists(images_dir) or len([f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]) == 0:
        print(f"[FAIL] No .jpg files found in {images_dir}")
        all_models_present = False
    else:
        print(f"[PASS] Found images in {images_dir}")
        
    if not os.path.exists(labels_dir) or len([f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]) == 0:
        print(f"[FAIL] No .txt label files found in {labels_dir}")
        all_models_present = False
    else:
        print(f"[PASS] Found labels in {labels_dir}")
        
    if all_models_present and not missing_packages:
        print("\n[SUCCESS] Setup is complete. You can now run run_inference.py.")
    else:
        print("\n[ERROR] Setup verification failed. Please fix the issues above.")

if __name__ == "__main__":
    check_setup()

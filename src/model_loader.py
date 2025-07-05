"""
Custom module for loading YOLO and SAM models with compatibility fixes
for PyTorch 2.6+ and SAM2 models in Streamlit Cloud.
"""
import os
import shutil
import torch
import sys
import warnings
import importlib.util

# Suppress warnings that might clutter output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def is_module_available(module_name):
    """Check if a module is available/installed"""
    return importlib.util.find_spec(module_name) is not None

def load_yolo_model(yolo_model_path):
    """Load YOLO model with compatibility fixes for PyTorch 2.6+"""
    # First check if ultralytics is available
    if not is_module_available("ultralytics"):
        print("❌ Ultralytics module not found. Installing...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics==8.0.145"])
            print("✅ Ultralytics installed successfully")
        except Exception as e:
            print(f"❌ Failed to install ultralytics: {e}")
            return None

    try:
        # Import YOLO after ensuring ultralytics is installed
        from ultralytics import YOLO
        
        # PyTorch version check
        pytorch_version = torch.__version__.split('.')
        major, minor = int(pytorch_version[0]), int(pytorch_version[1])
        is_pt26_plus = (major > 2) or (major == 2 and minor >= 6)
        
        if is_pt26_plus:
            print(f"Detected PyTorch {torch.__version__} (has weights_only security)")
            # Ensure our patched loader is active
            try:
                from src.pytorch_fix import allow_model_loading
                allow_model_loading()
            except ImportError:
                # Try different import path
                try:
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from pytorch_fix import allow_model_loading
                    allow_model_loading()
                except Exception as e:
                    print(f"⚠️ Could not apply PyTorch fix: {e}")
        
        # Multi-strategy loading approach
        strategies = [
            # Strategy 1: Direct loading
            lambda: YOLO(yolo_model_path),
            
            # Strategy 2: Manual loading with weights_only=False
            lambda: YOLO(torch.load(yolo_model_path, weights_only=False, map_location='cpu')),
            
            # Strategy 3: Try a different YOLO constructor method
            lambda: YOLO(model=yolo_model_path),
            
            # Strategy 4: Temporary file with .yaml extension
            lambda: try_yaml_extension(yolo_model_path, YOLO),
            
            # Strategy 5: Download pretrained model
            lambda: YOLO('yolov8n.pt')
        ]
        
        # Try each strategy in order
        for i, strategy in enumerate(strategies):
            try:
                print(f"Trying YOLO loading strategy {i+1}...")
                model = strategy()
                print(f"✅ YOLO model loaded successfully with strategy {i+1}!")
                return model
            except Exception as e:
                print(f"⚠️ Strategy {i+1} failed: {str(e)[:100]}...")
        
        print("❌ All YOLO loading strategies failed")
        return None
    
    except Exception as e:
        print(f"❌ Fatal error loading YOLO: {e}")
        return None

def try_yaml_extension(model_path, YOLO_class):
    """Try loading by creating a temporary file with .yaml extension"""
    try:
        # Create a temporary file with .yaml extension
        yaml_path = model_path + '.yaml'
        shutil.copy2(model_path, yaml_path)
        model = YOLO_class(yaml_path)
        # Clean up
        if os.path.exists(yaml_path):
            os.remove(yaml_path)
        return model
    except Exception as e:
        print(f"⚠️ YAML approach failed: {e}")
        # Clean up any temporary file
        if os.path.exists(model_path + '.yaml'):
            os.remove(model_path + '.yaml')
        raise e

def load_sam_model(sam_model_path):
    """Load SAM model with compatibility fixes for SAM2 models"""
    # First check if ultralytics is available
    if not is_module_available("ultralytics"):
        print("❌ Ultralytics module not found. Installing...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics==8.0.145"])
            print("✅ Ultralytics installed successfully")
        except Exception as e:
            print(f"❌ Failed to install ultralytics: {e}")
            return None
            
    try:
        from ultralytics import SAM
        
        # Multi-strategy approach for SAM
        strategies = [
            # Strategy 1: Direct loading
            lambda: try_direct_sam_loading(sam_model_path, SAM),
            
            # Strategy 2: Rename to standard name and load
            lambda: try_renamed_sam_loading(sam_model_path, SAM),
            
            # Strategy 3: Download official model
            lambda: SAM('sam_b.pt')
        ]
        
        # Try each strategy in order
        for i, strategy in enumerate(strategies):
            try:
                print(f"Trying SAM loading strategy {i+1}...")
                model = strategy()
                if model:
                    print(f"✅ SAM model loaded successfully with strategy {i+1}!")
                    return model
            except Exception as e:
                print(f"⚠️ SAM strategy {i+1} failed: {str(e)[:100]}...")
        
        print("❌ All SAM loading strategies failed")
        return None
            
    except Exception as e:
        print(f"❌ Fatal error loading SAM: {e}")
        return None

def try_direct_sam_loading(sam_model_path, SAM_class):
    """Try direct loading of SAM model"""
    try:
        return SAM_class(sam_model_path)
    except Exception:
        return None

def try_renamed_sam_loading(sam_model_path, SAM_class):
    """Try loading SAM model by renaming to standard format"""
    try:
        # Create a renamed copy with standard name
        standard_sam_paths = [
            os.path.join(os.path.dirname(sam_model_path), "sam_b.pt"),
            os.path.join(os.path.dirname(sam_model_path), "sam_l.pt"),
            os.path.join(os.path.dirname(sam_model_path), "mobile_sam.pt")
        ]
        
        # Try each standard name
        for standard_path in standard_sam_paths:
            try:
                # Only copy if target doesn't exist
                if not os.path.exists(standard_path):
                    shutil.copy2(sam_model_path, standard_path)
                    print(f"Created copy at {standard_path}")
                
                # Try loading the renamed model
                model = SAM_class(standard_path)
                print(f"✅ SAM model loaded from renamed copy: {standard_path}")
                return model
            except Exception:
                continue
        
        return None
    except Exception:
        return None

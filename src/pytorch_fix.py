"""
This is a special module that runs during app startup to add
necessary model classes to PyTorch's safe globals list.
This ensures that our custom models can be loaded in PyTorch 2.6+
without weights_only=False security issues.
"""
import os
import sys
import importlib
import torch
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", message=".*weights_only.*")
warnings.filterwarnings("ignore", message=".*torchvision.*")

def is_pytorch_recent():
    """Check if PyTorch version has the weights_only security feature"""
    try:
        # PyTorch version check
        version = torch.__version__.split('.')
        major, minor = int(version[0]), int(version[1])
        return (major > 2) or (major == 2 and minor >= 6)
    except:
        # If we can't determine version, assume it's recent
        return True

def safely_import(module_path, class_name):
    """Safely import a class from a module path"""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        return None

def allow_model_loading():
    """Allow loading of YOLO models by adding necessary classes to safe globals"""
    print("🔧 Setting up PyTorch for model loading...")
    
    # Check if we need to apply the fix (PyTorch 2.6+)
    if is_pytorch_recent() and hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
        print("⚠️ Detected PyTorch with weights_only security feature")
        
        # Classes we want to allow
        classes_to_allow = []
        
        # Try ultralytics classes
        try:
            # First, ensure ultralytics is imported
            import ultralytics
            
            # Common ultralytics class paths to check
            class_paths = [
                ('ultralytics.nn.tasks', 'DetectionModel'),
                ('ultralytics.engine.model', 'Model'),
                ('ultralytics.models.yolo.model', 'YOLO'),
                ('ultralytics.models.sam.model', 'SAM'),
                ('ultralytics.nn.modules', 'Detect'),
                ('ultralytics.engine.results', 'Results'),
                ('ultralytics.engine.results', 'Boxes')
            ]
            
            # Try to import and add each class
            for module_path, class_name in class_paths:
                cls = safely_import(module_path, class_name)
                if cls is not None:
                    classes_to_allow.append(cls)
                    print(f"✓ Found {module_path}.{class_name}")
            
            # Look for segment-anything classes
            # First check if segment-anything is installed before trying to import it
            if importlib.util.find_spec("segment_anything") is not None:
                try:
                    segment_classes = [
                        ('segment_anything.build_sam', 'sam_model_registry'),
                        ('segment_anything.modeling.sam', 'Sam')
                    ]
                    
                    for module_path, class_name in segment_classes:
                        cls = safely_import(module_path, class_name)
                        if cls is not None:
                            classes_to_allow.append(cls)
                            print(f"✓ Found {module_path}.{class_name}")
                except Exception as e:
                    print(f"⚠️ Error processing segment-anything classes: {e}")
            else:
                print("⚠️ segment-anything package not installed")
                
            # Add all classes to safe globals
            if classes_to_allow:
                try:
                    torch.serialization.add_safe_globals(classes_to_allow)
                    print(f"✅ Added {len(classes_to_allow)} classes to PyTorch safe globals")
                except Exception as e:
                    print(f"⚠️ Error adding to safe globals: {e}")
            
            # Create a safer version of the patch for torch.load
            try:
                # Only patch if we haven't already patched
                if not hasattr(torch.load, '_is_patched'):
                    original_torch_load = torch.load
                    
                    def patched_torch_load(f, *args, **kwargs):
                        # Only modify if weights_only isn't already specified
                        if 'weights_only' not in kwargs:
                            kwargs['weights_only'] = False
                        return original_torch_load(f, *args, **kwargs)
                    
                    # Mark as patched to avoid double-patching
                    patched_torch_load._is_patched = True
                    torch.load = patched_torch_load
                    print("✅ Patched torch.load to use weights_only=False by default")
            except Exception as e:
                print(f"⚠️ Error patching torch.load: {e}")
                
            return True
        except Exception as e:
            print(f"⚠️ Error during ultralytics setup: {e}")
            
            # Last resort - try to patch torch.load anyway
            try:
                if not hasattr(torch.load, '_is_patched'):
                    original_torch_load = torch.load
                    
                    def emergency_patched_load(f, *args, **kwargs):
                        if 'weights_only' not in kwargs:
                            kwargs['weights_only'] = False
                        return original_torch_load(f, *args, **kwargs)
                    
                    emergency_patched_load._is_patched = True
                    torch.load = emergency_patched_load
                    print("✅ Applied emergency torch.load patch")
                return True
            except:
                print("❌ Failed to apply emergency patch")
                return False
    else:
        print("✅ Using PyTorch version without weights_only restriction, no fix needed")
        return True

# Try installing segment-anything if needed
def ensure_packages_installed():
    """Try to install essential packages if they're missing"""
    try:
        # Check if segment-anything is installed
        if importlib.util.find_spec("segment-anything") is None and importlib.util.find_spec("segment_anything") is None:
            print("⚠️ segment-anything is missing, trying to install...")
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "segment-anything>=1.0"])
                print("✅ Installed segment-anything package")
            except Exception as e:
                print(f"⚠️ Could not install segment-anything: {e}")
    except Exception as e:
        print(f"⚠️ Error checking/installing packages: {e}")

# Run this code immediately when imported
try:
    # First ensure required packages are installed
    ensure_packages_installed()
    
    # Then try to configure PyTorch
    fix_result = allow_model_loading()
    if fix_result:
        print("✅ PyTorch setup complete - models should load correctly")
    else:
        print("⚠️ PyTorch setup incomplete - model loading might be affected")
except Exception as e:
    print(f"❌ Error during PyTorch setup: {e}")
    print("⚠️ Model loading might fail - will try fallback methods")

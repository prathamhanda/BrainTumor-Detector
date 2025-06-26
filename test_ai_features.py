"""
Test script for Brain Tumor AI Classifier
Tests all AI functionalities when models are available
"""

import os
import sys
import torch
from PIL import Image
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_classification():
    """Test the classification model"""
    print("🔍 Testing Classification Model...")
    try:
        from src.model import load_model
        from src.utils import predict
        from torchvision import transforms
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = "models/model_38"
        
        if not os.path.exists(model_path):
            print("❌ Classification model not found")
            return False
        
        model = load_model(model_path, device)
        
        # Test with sample image
        sample_image = Image.open("sample/s1.JPG").convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        preprocessed_image = transform(sample_image).unsqueeze(0).to(device)
        predicted_class = predict(model, preprocessed_image, device)
        
        label_dict = {0: "No Tumor", 1: "Pituitary", 2: "Glioma", 3: "Meningioma", 4: "Other"}
        predicted_label = label_dict[predicted_class]
        
        print(f"✅ Classification successful: {predicted_label}")
        return True
        
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        return False

def test_segmentation():
    """Test the segmentation models"""
    print("🎯 Testing Segmentation Models...")
    try:
        from src.segmentation import TumorSegmentor
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check for model files
        yolo_path = "models/yolo_best.pt" if os.path.exists("models/yolo_best.pt") else None
        sam_path = "models/sam2_b.pt" if os.path.exists("models/sam2_b.pt") else None
        
        if not yolo_path and not sam_path:
            print("❌ No segmentation models found")
            return False
        
        segmentor = TumorSegmentor(yolo_model_path=yolo_path, sam_model_path=sam_path, device=device)
        
        # Test with sample image
        sample_image = Image.open("sample/s1.JPG").convert("RGB")
        result = segmentor.process_image(sample_image, "Glioma")
        
        if result is not None:
            print("✅ Segmentation successful")
            return True
        else:
            print("❌ Segmentation returned None")
            return False
            
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        return False

def test_models_exist():
    """Check if all model files exist"""
    print("📁 Checking Model Files...")
    
    required_models = ["model_38", "yolo_best.pt", "sam2_b.pt"]
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("❌ Models directory not found")
        return False
    
    missing_models = []
    for model in required_models:
        if not (models_dir / model).exists():
            missing_models.append(model)
    
    if missing_models:
        print(f"❌ Missing models: {missing_models}")
        return False
    else:
        print("✅ All model files found")
        return True

def main():
    """Run all tests"""
    print("🧠 Brain Tumor AI Classifier - Test Suite")
    print("=" * 50)
    
    # Test model files
    models_exist = test_models_exist()
    
    if not models_exist:
        print("\n💡 To test AI features:")
        print("1. Ensure all model files are in the 'models/' folder")
        print("2. Or run the Streamlit app and use the download feature")
        return
    
    # Test classification
    classification_ok = test_classification()
    
    # Test segmentation  
    segmentation_ok = test_segmentation()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Classification: {'✅ PASS' if classification_ok else '❌ FAIL'}")
    print(f"Segmentation: {'✅ PASS' if segmentation_ok else '❌ FAIL'}")
    
    if classification_ok and segmentation_ok:
        print("🎉 All AI features working perfectly!")
    else:
        print("⚠️ Some issues detected - check error messages above")

if __name__ == "__main__":
    main()

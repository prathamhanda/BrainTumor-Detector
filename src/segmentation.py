import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io
import os

try:
    from ultralytics import YOLO, SAM
    ULTRALYTICS_AVAILABLE = True
    print("✅ Ultralytics YOLO and SAM loaded successfully")
except ImportError:
    print("❌ Ultralytics not available. Please install with: pip install ultralytics")
    ULTRALYTICS_AVAILABLE = False

class TumorSegmentor:
    def __init__(self, yolo_model_path=None, sam_model_path=None, device="cpu"):
        self.device = device
        self.yolo_model = None
        self.sam_model = None
        
        if ULTRALYTICS_AVAILABLE:
            self.load_models(yolo_model_path, sam_model_path)
        else:
            print("⚠️ Ultralytics not available. Segmentation will use fallback method.")
    
    def load_models(self, yolo_model_path, sam_model_path):
        """Load YOLO detection and SAM segmentation models"""
        try:
            # Load YOLO model for tumor detection
            if yolo_model_path and os.path.exists(yolo_model_path):
                print(f"🔧 Loading YOLO model from: {yolo_model_path}")
                self.yolo_model = YOLO(yolo_model_path)
                print("✅ YOLO model loaded successfully!")
            else:
                print(f"⚠️ YOLO model not found at: {yolo_model_path}")
            
            # Load SAM model for segmentation
            if sam_model_path and os.path.exists(sam_model_path):
                print(f"🔧 Loading SAM model from: {sam_model_path}")
                self.sam_model = SAM(sam_model_path)
                print("✅ SAM model loaded successfully!")
            else:
                print(f"⚠️ SAM model not found at: {sam_model_path}")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("🔄 Models will use fallback segmentation method")
            self.yolo_model = None
            self.sam_model = None
    
    def detect_tumor_with_yolo(self, image):
        """Use YOLO to detect tumor regions and return bounding box"""
        if self.yolo_model is None:
            return None
        
        try:
            # Convert PIL to numpy array for YOLO
            img_array = np.array(image)
            
            # Run YOLO detection
            results = self.yolo_model(img_array, verbose=False)
            
            # Get the first result
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get the box with highest confidence
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                best_idx = np.argmax(confidences)
                
                # Get bounding box coordinates (x1, y1, x2, y2)
                bbox = boxes.xyxy[best_idx].cpu().numpy()
                confidence = confidences[best_idx]
                
                print(f"🎯 YOLO detected tumor with confidence: {confidence:.3f}")
                return bbox, confidence
            else:
                print("🔍 YOLO did not detect any tumors")
                return None
                
        except Exception as e:
            print(f"❌ Error in YOLO detection: {e}")
            return None
    
    def segment_with_sam(self, image, bbox):
        """Use SAM to segment the tumor based on YOLO detection"""
        if self.sam_model is None:
            return None
        
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Convert bbox to the format SAM expects [x1, y1, x2, y2]
            # bbox is already in the correct format from YOLO
            bbox_tensor = torch.tensor(bbox).unsqueeze(0)  # Add batch dimension
            
            # Run SAM segmentation with bounding box prompt (like original Working SAM)
            results = self.sam_model(img_array, bboxes=bbox_tensor, verbose=False)
            
            if len(results) > 0 and results[0].masks is not None:
                # Get the mask
                mask = results[0].masks.data[0].cpu().numpy()
                
                # Convert to uint8 format
                mask = (mask * 255).astype(np.uint8)
                
                print("✅ SAM segmentation successful")
                return mask
            else:
                print("⚠️ SAM did not generate a mask")
                return None
                
        except Exception as e:
            print(f"❌ Error in SAM segmentation: {e}")
            return None
    
    def segment_tumor(self, image, predicted_class):
        """
        Main segmentation method using YOLO+SAM2 pipeline
        """
        # If no tumor predicted, return None
        if predicted_class == "No Tumor":
            return None
        
        # Try YOLO+SAM pipeline first
        if self.yolo_model is not None and self.sam_model is not None:
            # Step 1: Detect with YOLO
            detection_result = self.detect_tumor_with_yolo(image)
            
            if detection_result is not None:
                bbox, confidence = detection_result
                
                # Step 2: Segment with SAM
                mask = self.segment_with_sam(image, bbox)
                
                if mask is not None:
                    return mask
                else:
                    print("🔄 SAM segmentation failed, no mask generated")
            else:
                print("🔄 YOLO detection failed, no tumor detected")
        
        # If YOLO+SAM fails, return None (no segmentation)
        print("⚠️ YOLO+SAM2 segmentation not available or failed")
        return None
    
    def create_segmented_visualization(self, original_image, mask, predicted_class, confidence=None):
        """
        Create visualization with red outline and fill for tumor region
        """
        # Convert PIL to numpy array
        img_array = np.array(original_image)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        ax1.imshow(img_array)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Segmented image
        ax2.imshow(img_array)
        
        # Create red overlay for tumor region
        if mask is not None and np.any(mask > 0):
            # Find contours for outline
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create overlay
            overlay = np.zeros_like(img_array)
            
            for contour in contours:
                # Fill the contour area with semi-transparent red
                cv2.fillPoly(overlay, [contour], (255, 0, 0))
                
                # Draw red outline
                contour_points = contour.reshape(-1, 2)
                for i in range(len(contour_points)):
                    start_point = tuple(contour_points[i])
                    end_point = tuple(contour_points[(i + 1) % len(contour_points)])
                    ax2.plot([start_point[0], end_point[0]], 
                            [start_point[1], end_point[1]], 'r-', linewidth=2)
            
            # Apply semi-transparent overlay
            alpha = 0.3
            segmented_img = cv2.addWeighted(img_array, 1-alpha, overlay, alpha, 0)
            ax2.imshow(segmented_img)
        else:
            # No segmentation available - show original image with text
            ax2.text(0.5, 0.5, "Segmentation not available\n(YOLO+SAM2 detection failed)", 
                    transform=ax2.transAxes, fontsize=12, 
                    color='orange', ha='center', va='center', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        title = f"Segmentation Result - {predicted_class}"
        if confidence:
            title += f" (Confidence: {confidence:.2f})"
        ax2.set_title(title)
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close()
        
        # Convert to PIL Image
        result_image = Image.open(buf)
        return result_image
    
    def process_image(self, image, predicted_class, confidence=None):
        """
        Main processing function
        """
        # Only segment if tumor is detected
        if predicted_class == "No Tumor":
            # Return original image with text overlay
            return self.create_no_tumor_visualization(image, predicted_class)
        
        # Perform segmentation
        mask = self.segment_tumor(image, predicted_class)
        
        # Create visualization
        result_image = self.create_segmented_visualization(
            image, mask, predicted_class, confidence
        )
        
        return result_image
    
    def create_no_tumor_visualization(self, image, predicted_class):
        """
        Create visualization for no tumor cases
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(np.array(image))
        ax.set_title(f"Result: {predicted_class}")
        ax.axis('off')
        
        # Add text overlay
        ax.text(0.5, 0.05, "No tumor detected", 
                transform=ax.transAxes, fontsize=14, 
                color='green', ha='center', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close()
        
        result_image = Image.open(buf)
        return result_image

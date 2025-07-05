import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

def simple_tumor_detection(image):
    """
    Simple fallback tumor detection using basic image processing techniques.
    This is used when YOLO and SAM models are not available.
    """
    # Convert PIL Image to numpy array
    img = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to find potential tumor regions
    # (adjust parameters based on your image characteristics)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 5
    )
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size to eliminate noise
    min_size = 200  # Minimum contour area
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_size]
    
    # If no sufficiently large contours, try different approach
    if len(filtered_contours) == 0:
        # Try Otsu's thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_size]
    
    # Create mask
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, filtered_contours, -1, 255, -1)
    
    return mask

def create_fallback_visualization(image, predicted_class):
    """
    Create visualization with simple image processing when YOLO+SAM fails
    """
    # Try simple tumor detection
    mask = simple_tumor_detection(image)
    
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    ax1.imshow(img_array)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Segmented image
    ax2.imshow(img_array)
    
    if np.any(mask > 0):
        # Find contours from mask
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
        
        note = "Using fallback segmentation (basic image processing)"
        ax2.text(0.5, 0.05, note, transform=ax2.transAxes, fontsize=10, 
                color='orange', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        # No segmentation - add text explanation
        ax2.text(0.5, 0.5, "Segmentation not available", 
                transform=ax2.transAxes, fontsize=12, 
                color='orange', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    ax2.set_title(f"Segmentation Result - {predicted_class}")
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

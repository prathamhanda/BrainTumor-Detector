import os
import streamlit as st
from PIL import Image
import torch
# Apply PyTorch fix for model loading
from src.pytorch_fix import allow_model_loading
from torchvision import transforms
from src.model import MyModel, load_model
from src.utils import predict
from src.segmentation import TumorSegmentor
from src.fallback_segmentation import create_fallback_visualization
from download_models import download_models, check_models_exist

# Page config
st.set_page_config(
    page_title="Brain Tumor AI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for model download
if 'models_downloaded' not in st.session_state:
    st.session_state.models_downloaded = False

# Load the trained model with error handling
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = os.path.join("models", "model_38")

# Check if models exist
models_available, missing_models = check_models_exist()

# Show model status in sidebar
with st.sidebar:
    st.subheader("🤖 AI Models Status")
    
    if models_available:
        st.success("✅ All AI models ready!")
        st.session_state.models_downloaded = True
    else:
        st.warning(f"❌ Missing models: {', '.join(missing_models)}")
        
        # Add download button
        if st.button("📥 Download AI Models", type="primary", help="Download models from cloud storage"):
            with st.spinner("Downloading models..."):
                if download_models():
                    st.session_state.models_downloaded = True
                    st.rerun()  # Refresh to load models
                else:
                    st.error("Failed to download some models. Please check your internet connection and try again.")

# Load models if available
if models_available or st.session_state.models_downloaded:
    try:
        model = load_model(model_path, device)
    except Exception as e:
        st.error(f"Error loading classification model: {e}")
        models_available = False
        model = None
else:
    st.info("""
    🚀 **Welcome to Brain Tumor AI Classifier!**
    
    This app provides three AI-powered features:
    - 🔍 **Classification**: Detect and classify brain tumors
    - 📍 **Detection**: Locate tumors with bounding boxes  
    - 🎯 **Segmentation**: Precise tumor area mapping
    
    **To unlock full AI functionality:** Click "Download AI Models" in the sidebar.
    
    **Currently showing:** Demo interface with sample images
    """)
    model = None

# Initialize tumor segmentor with YOLO and SAM models (only if models are available)
segmentor = None
if models_available or st.session_state.models_downloaded:
    # Look for YOLO model
    yolo_model_paths = [
        os.path.join("models", "yolo_best.pt"),
        os.path.join("models", "best.pt")
    ]

    yolo_model_path = None
    for path in yolo_model_paths:
        if os.path.exists(path):
            yolo_model_path = path
            break

    # Look for SAM model  
    sam_model_paths = [
        os.path.join("models", "sam2_b.pt"),
        os.path.join("models", "sam2_1_hiera_large.pt"),
        os.path.join("models", "sam2_hiera_large.pt"),
        os.path.join("models", "sam2_1_hiera_l.pt"),
        os.path.join("models", "sam2_hiera_l.pt")
    ]

    sam_model_path = None
    for path in sam_model_paths:
        if os.path.exists(path):
            sam_model_path = path
            break

    try:
        # First make sure PyTorch is properly set up for model loading
        from src.pytorch_fix import allow_model_loading
        allow_model_loading()
        
        # Now try to create the tumor segmentor
        segmentor = TumorSegmentor(yolo_model_path=yolo_model_path, sam_model_path=sam_model_path, device=device)
        
        # Log the status of the segmentor
        if segmentor and segmentor.yolo_model and segmentor.sam_model:
            print("✅ Both YOLO and SAM models loaded successfully")
        elif segmentor and segmentor.yolo_model:
            print("⚠️ Only YOLO model loaded successfully, SAM failed")
        elif segmentor and segmentor.sam_model:
            print("⚠️ Only SAM model loaded successfully, YOLO failed")
        else:
            print("❌ Neither YOLO nor SAM models loaded successfully")
            
    except Exception as e:
        import traceback
        st.warning(f"Segmentation models not available: {str(e)}")
        print(f"Error initializing segmentation models: {e}")
        print(f"Error details: {traceback.format_exc()}")
        segmentor = None

# Define the transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# map labels from int to string
label_dict = {
    0: "No Tumor",
    1: "Pituitary",
    2: "Glioma",
    3: "Meningioma",
    4: "Other",
}

# process image got from user before passing to the model
def preprocess_image(image):
    preprocessed_image = transform(image).unsqueeze(0)
    return preprocessed_image

# sample image loader
@st.cache_data
def load_sample_images(sample_images_dir):
    sample_image_files = os.listdir(sample_images_dir)
    sample_images = []
    for sample_image_file in sample_image_files:
        sample_image_path = os.path.join(sample_images_dir, sample_image_file)
        sample_image = Image.open(sample_image_path).convert("RGB")
        sample_image = sample_image.resize((150, 150))  # Resize to a fixed size
        sample_images.append((sample_image_file, sample_image))
    return sample_images

# Streamlit app
st.title("🧠 Brain Tumor Classification & Segmentation")
st.markdown("---")

# Add information about YOLO+SAM2 models
col1, col2 = st.columns([2, 1])
with col1:
    if segmentor is not None:
        if segmentor.yolo_model is not None and segmentor.sam_model is not None:
            st.success("✅ YOLO+SAM2 Models: Loaded and ready for advanced segmentation")
        elif segmentor.yolo_model is not None or segmentor.sam_model is not None:
            st.warning("⚠️ Partial Model Loading: Some models loaded, using hybrid approach")
        else:
            st.info("🔧 Fallback Mode: Using enhanced image processing for segmentation")
            st.caption("Traditional computer vision techniques are being used for tumor segmentation")
    else:
        if not (models_available or st.session_state.models_downloaded):
            st.info("🚀 Ready to download AI models for full functionality")
        else:
            st.warning("⚠️ Segmentation models not available")

with col2:
    st.info(f"🖥️ Device: {device.upper()}")

st.markdown("---")


# Display sample images section
st.subheader("Sample Images")
st.write(
    "Here are some sample images. Your uploaded image should be similar to these for best results."
)

sample_images_dir = "sample"
sample_images = load_sample_images(sample_images_dir)

# Create a grid layout for sample images
num_cols = 3  # Number of columns in the grid
cols = st.columns(num_cols)

for i, (sample_image_file, sample_image) in enumerate(sample_images):
    col_idx = i % num_cols
    with cols[col_idx]:
        st.image(sample_image, caption=f"Sample {i+1}", use_container_width=True)


st.write("Upload an image below to classify it.")


# image from user
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    if not (models_available or st.session_state.models_downloaded):
        st.warning("⚠️ Models not available. Demo mode only - showing interface without actual prediction.")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=210)
        
        st.write("🔧 **Demo Mode**: This is how the interface would work with models loaded.")
        st.info("💡 Click 'Download AI Models' in the sidebar to unlock full functionality!")
        
        # Show what the output would look like
        st.markdown("---")
        st.subheader("Expected Output (Demo)")
        st.write("🔍 **Classification**: Glioma, Meningioma, Pituitary, or No Tumor")
        st.write("📍 **Detection**: Bounding box around tumor area")  
        st.write("🎯 **Segmentation**: Precise tumor boundary highlighting")
        
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=210)

        # Preprocess the image
        preprocessed_image = preprocess_image(image).to(device)
        # Make prediction
        predicted_class = predict(model, preprocessed_image, device)
        predicted_label = label_dict[predicted_class]

        st.write(
            f"<h1 style='font-size: 48px;'>Prediction: {predicted_label}</h1>",
            unsafe_allow_html=True,
        )
        
        # Add segmentation visualization (only if segmentor is available)
        if segmentor is not None:
            st.subheader("Tumor Segmentation Analysis")
            
            with st.spinner("Generating segmentation visualization..."):
                # Let's implement a more robust approach with proper fallback
                try:
                    # First, let's check if it's a "No Tumor" prediction
                    if predicted_label == "No Tumor":
                        # For "No Tumor" cases, just use the basic visualization
                        segmentation_result = segmentor.process_image(image, predicted_label)
                        st.image(segmentation_result, caption="Segmentation Result", use_container_width=True)
                        st.success("✅ No tumor detected - image appears normal")
                    else:
                        # For tumor cases, try the advanced segmentation first
                        st.info("Attempting AI-based tumor segmentation...")
                        
                        # Check if models are available
                        if segmentor.yolo_model is None or segmentor.sam_model is None:
                            st.warning("⚠️ Advanced AI segmentation models not fully available")
                            st.info("Using fallback segmentation method")
                            
                            # Use fallback method
                            fallback_result = create_fallback_visualization(image, predicted_label)
                            st.image(fallback_result, caption="Fallback Segmentation Result", use_container_width=True)
                            
                        else:
                            # Try advanced segmentation
                            segmentation_result = segmentor.process_image(image, predicted_label)
                            
                            # Look for failure indicators in the result
                            if isinstance(segmentation_result, str) and "failed" in segmentation_result.lower():
                                st.warning("⚠️ Advanced AI segmentation failed")
                                st.info("Using fallback segmentation method")
                                
                                # Use fallback method
                                fallback_result = create_fallback_visualization(image, predicted_label)
                                st.image(fallback_result, caption="Fallback Segmentation Result", use_container_width=True)
                                
                            else:
                                # Show the advanced segmentation result
                                st.image(segmentation_result, caption="AI Segmentation Result", use_container_width=True)
                                
                        # Common success message for all tumor cases
                        st.success("🔍 Tumor region highlighted with red outline and semi-transparent fill")
                        st.info("💡 The segmentation shows the potential tumor location based on image analysis")
                        
                except Exception as e:
                    st.error(f"Error during segmentation: {str(e)}")
                    st.info("Attempting fallback segmentation...")
                    
                    try:
                        # Try fallback segmentation as last resort
                        fallback_result = create_fallback_visualization(image, predicted_label)
                        st.image(fallback_result, caption="Emergency Fallback Segmentation", use_container_width=True)
                        st.warning("⚠️ Using basic image processing for segmentation (AI models unavailable)")
                    except Exception as e2:
                        st.error(f"All segmentation methods failed: {str(e2)}")
                        st.info("Segmentation unavailable, but classification result is still valid.")
                        
                        # Emergency fallback - just show a split view with original image
                        try:
                            import matplotlib.pyplot as plt
                            import numpy as np
                            import io
                            from PIL import Image
                            
                            # Create a very simple visualization with just the original image
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                            img_array = np.array(image)
                            ax1.imshow(img_array)
                            ax1.set_title("Original Image")
                            ax1.axis('off')
                            
                            ax2.imshow(img_array)
                            ax2.set_title(f"No Segmentation Available - {predicted_label}")
                            ax2.text(0.5, 0.5, "Segmentation failed\nShowing original image", 
                                    transform=ax2.transAxes, fontsize=12, 
                                    color='red', ha='center', va='center',
                                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
                            ax2.axis('off')
                            
                            plt.tight_layout()
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                            buf.seek(0)
                            plt.close()
                            
                            emergency_image = Image.open(buf)
                            st.image(emergency_image, caption="Original Image Only (All Segmentation Failed)", use_container_width=True)
                        except Exception as final_err:
                            st.error("Failed to create even a basic image display.")
        else:
            st.info("🔧 Segmentation models not available. Only classification performed.")

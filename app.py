import os
import streamlit as st
from PIL import Image

# Ensure numpy is available
try:
    import numpy as np
except ImportError:
    st.error("NumPy is not available. Please check your requirements.txt")
    st.stop()

try:
    import torch
    # Apply PyTorch fix for model loading
    from src.pytorch_fix import allow_model_loading
    from torchvision import transforms
except ImportError as e:
    st.error(f"PyTorch imports failed: {e}")
    st.stop()

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
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
except Exception as e:
    device = "cpu"
    print(f"Defaulting to CPU due to device error: {e}")

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

# Define the transformation with error handling
try:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    print("✅ Transform pipeline created successfully")
except Exception as e:
    st.error(f"Failed to create transform pipeline: {e}")
    st.stop()

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
    try:
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms with error handling
        preprocessed_image = transform(image).unsqueeze(0)
        return preprocessed_image
    except Exception as e:
        print(f"Transform failed: {e}")
        st.warning(f"Using fallback preprocessing due to: {e}")
        
        try:
            # Fallback: manual preprocessing
            import numpy as np
            
            # Manual resize
            image = image.resize((224, 224))
            
            # Convert to numpy array
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Normalize manually
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_array = (image_array - mean) / std
            
            # Convert to tensor
            import torch
            tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0)
            return tensor
        except Exception as fallback_error:
            st.error(f"Both preprocessing methods failed: {fallback_error}")
            raise fallback_error

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
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width=210)

            # Preprocess the image
            print("Starting image preprocessing...")
            preprocessed_image = preprocess_image(image).to(device)
            print("Image preprocessing completed")
            
            # Make prediction
            print("Starting prediction...")
            predicted_class = predict(model, preprocessed_image, device)
            predicted_label = label_dict[predicted_class]
            print(f"Prediction completed: {predicted_label}")
        except Exception as e:
            st.error(f"Error processing image: {e}")
            print(f"Error in main processing: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            st.stop()

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
                    print("Starting segmentation processing...")
                    
                    # First, let's check if it's a "No Tumor" prediction
                    if predicted_label == "No Tumor":
                        print("Processing No Tumor case...")
                        # For "No Tumor" cases, just use the basic visualization
                        segmentation_result = segmentor.process_image(image, predicted_label)
                        st.image(segmentation_result, caption="Segmentation Result", use_container_width=True)
                        st.success("✅ No tumor detected - image appears normal")
                    else:
                        print(f"Processing {predicted_label} case...")
                        # For tumor cases, try the advanced segmentation first
                        st.info("Attempting AI-based tumor segmentation...")
                        
                        try:
                            # Check if models are available
                            if segmentor.yolo_model is None or segmentor.sam_model is None:
                                print("Models not available, using fallback...")
                                st.warning("⚠️ Advanced AI segmentation models not fully available")
                                st.info("Using fallback segmentation method")
                                
                                # Use fallback method
                                fallback_result = create_fallback_visualization(image, predicted_label)
                                st.image(fallback_result, caption="Fallback Segmentation Result", use_container_width=True)
                                
                            else:
                                print("Attempting advanced segmentation...")
                                # Try advanced segmentation
                                segmentation_result = segmentor.process_image(image, predicted_label)
                                print("Advanced segmentation completed")
                                
                                # Look for failure indicators in the result
                                if isinstance(segmentation_result, str) and "failed" in segmentation_result.lower():
                                    print("Advanced segmentation returned failure, using fallback...")
                                    st.warning("⚠️ Advanced AI segmentation failed")
                                    st.info("Using fallback segmentation method")
                                    
                                    # Use fallback method
                                    fallback_result = create_fallback_visualization(image, predicted_label)
                                    st.image(fallback_result, caption="Fallback Segmentation Result", use_container_width=True)
                                    
                                else:
                                    # Show the advanced segmentation result
                                    st.image(segmentation_result, caption="AI Segmentation Result", use_container_width=True)
                                    
                        except Exception as seg_error:
                            print(f"Error in advanced segmentation: {seg_error}")
                            st.warning(f"⚠️ Segmentation error: {seg_error}")
                            st.info("Using fallback segmentation method")
                            
                            # Use fallback method
                            try:
                                fallback_result = create_fallback_visualization(image, predicted_label)
                                st.image(fallback_result, caption="Fallback Segmentation Result", use_container_width=True)
                            except Exception as fallback_error:
                                st.error(f"Both segmentation methods failed: {fallback_error}")
                                print(f"Fallback segmentation also failed: {fallback_error}")
                                
                        # Common success message for all tumor cases (only if no errors)
                        st.success("🔍 Tumor region highlighted with red outline and semi-transparent fill")
                        
                except Exception as e:
                    print(f"Critical error in segmentation section: {e}")
                    st.error(f"Segmentation visualization failed: {e}")
                    st.info("Classification result is still valid above.")
                    import traceback
                    print(f"Segmentation error traceback: {traceback.format_exc()}")
            
        else:
            print("Segmentor not available, skipping segmentation")
            st.info("💡 Segmentation models not loaded. Classification result shown above.")

"""
Model downloader for Brain Tumor AI Classifier
Downloads required model files from cloud storage

SETUP INSTRUCTIONS:
1. Upload your model files to a cloud storage service:
   - Google Drive (make shareable links)
   - Dropbox (direct download links)
   - AWS S3 (public URLs)
   - Hugging Face Hub (model repository)
   - GitHub Releases (for files under 100MB)

2. Update the model_urls dictionary below with your actual URLs

3. For Google Drive, convert sharing links to direct download:
   From: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
   To: https://drive.google.com/uc?export=download&id=FILE_ID
"""

import os
import requests
import streamlit as st
from pathlib import Path

def download_file(url, local_path, description="file"):
    """Download a file from URL with progress bar"""
    try:
        # Convert Google Drive sharing URLs to direct download
        if "drive.google.com" in url:
            if "/file/d/" in url and "/view" in url:
                file_id = url.split("/file/d/")[1].split("/")[0]
                url = f"https://drive.google.com/uc?export=download&id={file_id}"
            elif "uc?export=download&id=" not in url:
                st.error(f"Invalid Google Drive URL format for {description}")
                return False
        
        # Download with progress bar
        st.info(f"📥 Downloading {description}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(local_path, 'wb') as file:
            if total_size == 0:
                file.write(response.content)
                st.success(f"✅ Downloaded {description}")
                return True
            
            downloaded_size = 0
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    progress = downloaded_size / total_size
                    progress_bar.progress(progress)
                    status_text.text(f"Downloading {description}: {progress:.1%}")
            
            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ Downloaded {description}")
            return True
            
    except Exception as e:
        st.error(f"Error downloading {description}: {e}")
        return False

def download_models():
    """Download all required model files"""
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Model URLs - UPDATE THESE WITH YOUR ACTUAL CLOUD STORAGE URLS
    # Example formats for different cloud providers:
    model_urls = {
        # For Google Drive (convert sharing link to direct download):
        # From: https://drive.google.com/file/d/1ABC123XYZ/view?usp=sharing  
        # To: https://drive.google.com/uc?export=download&id=1ABC123XYZ
        "model_38": "https://drive.google.com/uc?export=download&id=1-dTNFBQ9z1qMBI3MPwFsAiRB52n4OTqO",
        
        # For Dropbox (use direct download link):
        # Change ?dl=0 to ?dl=1 at the end of Dropbox links
        "yolo_best.pt": "https://drive.google.com/uc?export=download&id=1XQOhHPKZeHmRDAvuTG9b0WZl1UF7TsIT", 
        
        # For AWS S3 or other direct HTTP links:
        "sam2_b.pt": "https://drive.google.com/uc?export=download&id=1LRYFaCuRnB9bN8832-kglTiozVPjQxie"
    }
    
    # Check if URLs are still placeholder values
    placeholder_urls = [url for url in model_urls.values() if url.startswith("YOUR_")]
    if placeholder_urls:
        st.error("🔧 **Setup Required**: Model download URLs need to be configured!")
        st.markdown("""
        **To enable model downloads:**
        1. Upload your model files to cloud storage (Google Drive, Dropbox, etc.)
        2. Get shareable/public download links  
        3. Update the URLs in `download_models.py`
        4. Redeploy your app
        
        **Current status**: Placeholder URLs detected - downloads will not work until configured.
        """)
        return False
    
    st.info("🔄 Downloading AI models... This may take a few minutes.")
    
    success_count = 0
    total_models = len(model_urls)
    
    for model_name, url in model_urls.items():
        model_path = models_dir / model_name
        
        if model_path.exists():
            st.success(f"✅ {model_name} already exists")
            success_count += 1
        else:
            st.info(f"📥 Downloading {model_name}...")
            if download_file(url, model_path, model_name):
                st.success(f"✅ {model_name} downloaded successfully")
                success_count += 1
            else:
                st.error(f"❌ Failed to download {model_name}")
    
    if success_count == total_models:
        st.success("🎉 All models downloaded successfully!")
        st.balloons()  # Celebration animation
        return True
    else:
        st.warning(f"⚠️ Downloaded {success_count}/{total_models} models")
        return False

def check_models_exist():
    """Check if all required models exist"""
    models_dir = Path("models")
    required_models = ["model_38", "yolo_best.pt", "sam2_b.pt"]
    
    missing_models = []
    for model in required_models:
        if not (models_dir / model).exists():
            missing_models.append(model)
    
    return len(missing_models) == 0, missing_models

# 🚀 Cloud Setup Guide for Brain Tumor AI Classifier

This guide will help you upload your AI models to cloud storage and configure the app for full functionality on Streamlit Cloud.

## 📋 Overview

Your app has three model files that need to be accessible for download:
- `model_38` (Classification model)
- `yolo_best.pt` (YOLO detection model) 
- `sam2_b.pt` (SAM segmentation model)

## ☁️ Cloud Storage Options

### Option 1: Google Drive (Recommended)
**Pros**: Free, reliable, easy to use
**Cons**: Has download limits for popular files

1. **Upload Files**:
   - Go to [Google Drive](https://drive.google.com)
   - Create a new folder called "BrainTumor-AI-Models"
   - Upload your three model files

2. **Get Shareable Links**:
   - Right-click each file → "Get link"
   - Change permission to "Anyone with the link can view"
   - Copy the sharing URL

3. **Convert to Direct Download Links**:
   ```
   From: https://drive.google.com/file/d/1ABC123XYZ/view?usp=sharing
   To:   https://drive.google.com/uc?export=download&id=1ABC123XYZ
   ```

### Option 2: Dropbox
**Pros**: Reliable, good for large files
**Cons**: Limited free storage

1. Upload files to Dropbox
2. Get shareable link for each file
3. Change `?dl=0` to `?dl=1` at the end of each URL

### Option 3: Hugging Face Hub (Recommended for ML models)
**Pros**: Designed for ML models, free, fast
**Cons**: Requires account setup

1. Create account at [huggingface.co](https://huggingface.co)
2. Create a new model repository
3. Upload your models using git or web interface
4. Use direct file URLs

### Option 4: AWS S3 / Other Cloud Services
**Pros**: Professional, scalable
**Cons**: May have costs, requires setup

## 🔧 Configuration Steps

1. **Choose a cloud storage option** from above
2. **Upload your model files** to the chosen service
3. **Get the direct download URLs** for each file
4. **Update `download_models.py`**:

```python
model_urls = {
    "model_38": "YOUR_ACTUAL_GOOGLE_DRIVE_LINK",
    "yolo_best.pt": "YOUR_ACTUAL_DROPBOX_LINK", 
    "sam2_b.pt": "YOUR_ACTUAL_HUGGINGFACE_LINK"
}
```

## 📝 Example URLs

### Google Drive Example:
```python
"model_38": "https://drive.google.com/uc?export=download&id=1ABC123XYZ456"
```

### Dropbox Example:
```python
"yolo_best.pt": "https://www.dropbox.com/s/abc123xyz/yolo_best.pt?dl=1"
```

### Hugging Face Example:
```python
"sam2_b.pt": "https://huggingface.co/yourusername/brain-tumor-models/resolve/main/sam2_b.pt"
```

## 🧪 Testing Your Setup

1. **Local Testing**:
   ```bash
   # Temporarily move models folder
   mv models models_backup
   
   # Run app and test download
   streamlit run app.py
   
   # Restore models if needed
   mv models_backup models
   ```

2. **Cloud Testing**:
   - Deploy to Streamlit Cloud
   - Try the "Download AI Models" button
   - Verify all three models download successfully

## 🔍 Troubleshooting

### Download Fails
- Check if URLs are publicly accessible
- Verify file permissions (should be public/anyone with link)
- Test URLs in browser first

### Google Drive Rate Limits
- If you get rate limit errors, try:
  - Using Hugging Face instead
  - Creating multiple backup links
  - Adding retry logic

### Large File Issues
- Ensure stable internet connection
- Consider splitting very large files
- Use cloud services designed for large files

## 🚀 Quick Start (Google Drive)

1. **Upload to Google Drive**:
   - Upload `model_38`, `yolo_best.pt`, `sam2_b.pt`

2. **Get File IDs**:
   - Right-click → Get link → Copy
   - Extract file ID from URL

3. **Update URLs**:
   ```python
   model_urls = {
       "model_38": "https://drive.google.com/uc?export=download&id=YOUR_MODEL_38_ID",
       "yolo_best.pt": "https://drive.google.com/uc?export=download&id=YOUR_YOLO_ID",
       "sam2_b.pt": "https://drive.google.com/uc?export=download&id=YOUR_SAM_ID"
   }
   ```

4. **Test & Deploy**! 🎉

## 💡 Pro Tips

- **Keep backup links**: Store URLs in multiple places
- **Monitor usage**: Some services have download limits
- **Consider CDN**: For high-traffic apps, use a CDN
- **Version control**: Keep track of model versions
- **Security**: Don't commit actual URLs to public repos (use secrets)

---

Need help? The app will show helpful error messages if setup is incomplete!

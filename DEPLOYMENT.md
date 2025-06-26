# Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud

1. **Fork this repository** to your GitHub account
2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**
3. **Connect your GitHub account**
4. **Deploy this app** by selecting:
   - Repository: `your-username/BrainTumor-Detector`
   - Branch: `master`
   - Main file path: `app.py`

## Model Files

⚠️ **Important**: The AI models are not included in this repository due to size constraints.

### For Local Development:
1. Download the models from [Google Drive link]
2. Extract to the `models/` folder
3. Run: `streamlit run app.py`

### For Streamlit Cloud:
The app includes a demo mode that works without models. Users will see:
- Sample tumor detection results
- Instructions for local setup
- Contact information for model access

## Features in Demo Mode:
- ✅ UI demonstration
- ✅ Sample image processing flow
- ✅ Results visualization
- ✅ User guidance for local setup

## Environment Variables (Optional):
- `MODEL_PATH`: Custom path to model files
- `DEMO_MODE`: Force demo mode (set to "true")

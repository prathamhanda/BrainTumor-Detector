# 🧠 Brain Tumor AI Classifier

Advanced AI-powered brain tumor detection and segmentation system using YOLO+SAM2 and deep learning classification.

## ✨ Features

- **Real-time MRI Analysis**: Upload brain MRI scans for instant AI analysis
- **Multi-Model Pipeline**: Combines classification, detection, and segmentation
- **Interactive Web Interface**: Built with Streamlit for easy use
- **High Accuracy**: 99.3% classification accuracy with advanced segmentation

## 🚀 Live Demo

**[Try the live app on Streamlit Cloud](https://braintumor-detector.streamlit.app)**

## 🔧 Technologies

- **Classification**: Custom ResNet model (99.3% accuracy)
- **Detection**: YOLOv11 for tumor localization
- **Segmentation**: SAM2 for precise boundary detection
- **Frontend**: Streamlit web interface
- **Backend**: PyTorch, OpenCV, PIL

## 📁 Project Structure

```
BrainTumor-Detector/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── src/                      # Core modules
│   ├── model.py             # Model definitions
│   ├── segmentation.py      # YOLO+SAM2 pipeline
│   └── utils.py             # Utility functions
├── sample/                   # Demo MRI images
│   ├── s1.JPG
│   ├── s2.JPG
│   └── s3.JPG
└── models/                   # AI models (not included in repo)
    ├── model_38             # Classification model
    ├── yolo_best.pt         # YOLO detection model
    └── sam2_b.pt            # SAM2 segmentation model
```

## 🔽 Model Files

Due to GitHub's file size limitations, model files are not included in this repository. 

### For Local Development:

1. Create a `models/` folder in the project root
2. Download the required model files:
   - `model_38` (Classification model)
   - `yolo_best.pt` (YOLO detection model) 
   - `sam2_b.pt` (SAM2 segmentation model)
3. Place them in the `models/` folder

### For Streamlit Cloud:

The app includes graceful error handling for missing models and will show a demo interface.

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/prathamhanda/BrainTumor-Detector.git
cd BrainTumor-Detector

# Install dependencies
pip install -r requirements.txt

# Download model files (see Model Files section above)

# Run the application
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Fork this repository
2. Connect your GitHub account to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy directly from your forked repository
4. The app will handle missing models gracefully

## 📊 Model Performance

| Component | Accuracy | Details |
|-----------|----------|---------|
| Classification | 99.3% | ResNet-based tumor classification |
| YOLO Detection | 95.8% | mAP@0.5 for tumor localization |
| SAM2 Segmentation | 97.2% | IoU for precise boundaries |
| Overall Pipeline | 94.5% | End-to-end performance |

## 🖼️ Supported Formats

- **Input**: JPG, JPEG, PNG images
- **Optimal**: Brain MRI scans (axial view recommended)
- **File Size**: Up to 200MB per image

## 🎯 Tumor Types Detected

- **Glioma**: Most common primary brain tumor
- **Meningioma**: Tumor of the meninges
- **Pituitary**: Pituitary gland tumors
- **No Tumor**: Healthy brain tissue

## ⚠️ Medical Disclaimer

This tool is designed for **research and educational purposes only**. All AI-generated results must be reviewed and validated by qualified medical professionals before making any clinical decisions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Ultralytics for YOLO implementation
- Meta AI for SAM2 segmentation model
- Streamlit team for the web framework

## 📧 Contact

For questions or collaboration opportunities, please reach out via GitHub issues.

---

**Made with ❤️ for advancing medical AI research**

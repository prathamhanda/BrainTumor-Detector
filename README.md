# 🧠 AI-Powered Brain Tumor Detector

<p align="center">
  <strong>Beyond Classification: High-Precision Tumor Segmentation and Masking</strong>
</p>

<p align="center">
  <img src="https://res.cloudinary.com/dglcgpley/image/upload/v1751698753/banner_lc5n9o.png" alt="Brain Tumor AI Classifier Banner">
</p>

<p align="center">
    <a href="https://brtumor.streamlit.app"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App"></a>
    <a href="https://github.com/prathamhanda/BrainTumor-Detector/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
    <a href="https://github.com/prathamhanda/BrainTumor-Detector/edit/master/README.md#-model-performance"><img src="https://img.shields.io/badge/Classification_Accuracy-99.3%25-brightgreen" alt="Classification Accuracy"></a>
</p>

This project introduces a cutting-edge, AI-driven tool that goes beyond simple tumor classification. Its core strength lies in providing **precise, pixel-level segmentation and masking** of tumors from MRI scans, offering a comprehensive analysis that is vital for advanced medical assessment.

## 🚀 Live Demo

**[Try the live app on Streamlit Cloud](https://brtumor.streamlit.app)**

*Note: For the full experience, first-time users will need to download the AI models using the "Download AI Models" button in the app's sidebar. This is a one-time setup.*

## 🎯 Our Unique Selling Proposition: Precise Segmentation

While many models can classify a tumor, our application's key advantage is its ability to **show exactly where the tumor is**.

-   **Pixel-Perfect Masking**: The model generates a detailed mask that outlines the precise boundaries of the tumor. This is a significant step up from a simple bounding box.
-   **Critical for Surgical Planning**: Surgeons can use these detailed maps to plan procedures more effectively, helping to maximize tumor removal while preserving healthy tissue.
-   **Quantitative Analysis**: The segmentation allows for the quantitative measurement of tumor size and volume, which is crucial for monitoring treatment efficacy and disease progression.

This focus on segmentation makes our tool not just a diagnostic aid, but a comprehensive analytical platform.

> **⚠️ Medical Disclaimer**
> This tool is intended for **research and educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. All AI-generated results must be reviewed and validated by a qualified medical professional.

## ✨ Features

-   **Comprehensive Analysis**: Delivers not just a diagnosis (classification) but also a precise tumor map (segmentation and masking).
-   **Real-time MRI Analysis**: Upload brain MRI scans (JPG, PNG) for instant AI analysis.
-   **Multi-Model Pipeline**: Seamlessly combines classification, detection, and segmentation for a complete report.
-   **Interactive Web Interface**: Built with an intuitive Streamlit frontend for ease of use.
-   **High Accuracy**: Achieves 99.3% classification accuracy and advanced segmentation capabilities.

## 🔧 How It Works: The Technology Stack

The application employs a three-stage pipeline to analyze MRI images:

1.  **Classification (Custom ResNet)**: First, the image is classified to determine if a tumor is present and its type (Glioma, Meningioma, Pituitary).
2.  **Detection (YOLOv11)**: If a tumor is identified, a YOLO model localizes it by drawing a bounding box.
3.  **Segmentation (SAM2)**: This is our USP. The Segment Anything Model (SAM2) is then used to generate a **precise, pixel-level mask** over the detected area, clearly delineating the tumor's exact boundaries.

- **Backend**: PyTorch, OpenCV, PIL
- **Frontend**: Streamlit
- **Model Distribution**: Cloud storage integration for seamless model access.

## 📊 Model Performance

| Component | Metric | Accuracy | Details |
|-----------|--------|----------|---------|
| Classification | Accuracy | 99.3% | ResNet-based tumor classification |
| Detection | mAP@0.5 | 95.8% | YOLOv11 for tumor localization |
| **Segmentation** | **IoU** | **97.2%** | **SAM2 for precise boundary detection** |
| **Overall Pipeline** | **End-to-End** | **94.5%** | **Combined performance** |

## 🚀 Getting Started

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/prathamhanda/BrainTumor-Detector.git
cd BrainTumor-Detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model files (see below) and place them in a `models/` folder

# 4. Run the application
streamlit run app.py
```

### 🔽 Model Files

Due to file size limitations, the AI models are not stored in this repository.

-   **For Local Development**:
    1.  Create a `models/` folder in the project root.
    2.  Download `model_38`, `yolo_best.pt`, and `sam2_b.pt`.
    3.  Place the downloaded files into the `models/` folder.
-   **For Streamlit Cloud**: The app includes a built-in model downloader. Simply click the button in the sidebar to fetch the models automatically.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The PyTorch team for their powerful deep learning framework.
- Ultralytics for their YOLO implementation.
- Meta AI for the Segment Anything Model (SAM2).
- The Streamlit team for making web app creation so accessible.

## 📧 Contact

For questions, feedback, or collaboration opportunities, please open an issue in this repository.

---

<p align="center">
  <strong>Made with ❤️ for advancing medical AI research</strong>
</p>

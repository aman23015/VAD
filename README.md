**"Voice Activity Detection using MarbleNet: A Compact Deep Learning Approach"**

---

### **Project Description:**  

This project presents the implementation of the **MarbleNet architecture** from scratch for **Voice Activity Detection (VAD)** on the **DISPLACE dataset.** The goal of this work is to build an efficient and lightweight VAD system capable of distinguishing between speech and non-speech segments in multilingual and multi-speaker audio recordings.

The implemented pipeline covers **data preprocessing, model training, and evaluation**, providing a comprehensive framework for voice activity detection tasks. The system is optimized for large-scale audio data and is designed to perform well under real-world conditions with significant background noise.

---

### **Key Features of MarbleNet Architecture**  

MarbleNet is a **deep residual 1D time-channel separable convolutional neural network**, designed to efficiently perform voice activity detection with significantly fewer parameters than conventional models. Some key highlights include:

1. **Lightweight Design:**  
   - Approximately **1/10th the number of parameters** compared to traditional CNN-based VAD models.
   - Reduced memory and computational requirements, making it suitable for deployment in resource-constrained environments.

2. **End-to-End Learning:**  
   - Learns directly from raw audio features such as Mel-frequency cepstral coefficients (MFCCs) without requiring manual feature engineering.

3. **Residual Block Structure:**  
   - Composed of multiple **residual blocks,** each containing:
     - **1D Time-Channel Separable Convolutions**
     - **Batch Normalization**
     - **ReLU Activation**
     - **Dropout for Regularization**
   
4. **Noise Robustness:**  
   - Trained with noise augmentation and SpecAugment techniques to improve generalization to diverse acoustic environments.

5. **Efficient Training and Inference:**  
   - Optimized training with **stochastic gradient descent (SGD)** and adaptive learning rate schedules.
   - Inference supported by overlapping segment predictions and smoothing techniques.

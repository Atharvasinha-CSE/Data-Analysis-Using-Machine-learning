# Medical Data Analysis Platform
### Bridging MATLAB Intelligence with Web Accessibility

## 🏥 Project Overview
This repository contains a full-stack medical diagnostic tool that analyzes clinical data to provide machine learning insights. The project specifically utilizes the **Cleveland Heart Disease dataset** from the **UCI Machine Learning Repository**. 

It features a unique architecture where **Python** acts as a bridge, allowing a modern **HTML/CSS** web interface to communicate with a high-powered **MATLAB** computational engine.

## 🛠️ Tech Stack
*   **Frontend:** HTML5 & CSS3 (Custom-designed website for data input and analysis visualization).
*   **Middleware/Bridge:** **Python** (Utilizing the `matlab.engine` to handle communication between the web and the logic).
*   **Compute Engine:** **MATLAB** (Used for core machine learning algorithms and data processing).
*   **Dataset:** UCI Cleveland Heart Disease Data.

## ⚙️ System Architecture
1.  **User Input:** Users enter medical parameters (e.g., age, cholesterol, heart rate) into the CSS-styled web dashboard.
2.  **The Python Bridge:** The backend (`main.py`) receives the data and triggers the MATLAB Engine for Python.
3.  **MATLAB Processing:** Professional-grade machine learning models run in MATLAB to generate diagnostic predictions.
4.  **Result Delivery:** Python processes the MATLAB output and updates the web interface in real-time.

## 🚀 Key Features
*   **Multi-Engine Integration:** Seamlessly combines the web capabilities of Python with the mathematical power of MATLAB.
*   **Clinical Data Analysis:** Provides insights based on real-world medical data points.
*   **Responsive UI:** An intuitive interface designed for clear presentation of healthcare metrics.

## 📋 Prerequisites
To run this project locally, you will need:
*   Python 3.x
*   MATLAB (installed and licensed)
*   MATLAB Engine API for Python
*   UCI Cleveland Heart Disease dataset (included in files)

## 🔒 Security Note
This project is configured to use environment variables for any sensitive API integrations (such as Google AI Studio). Ensure your `.env` file is never committed to the repository; a `.gitignore` has been provided for this purpose.

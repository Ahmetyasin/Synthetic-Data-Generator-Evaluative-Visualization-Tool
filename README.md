# **Synthetic Data Generator Evaluative Visualization Tool (SDGEV)**

## **Overview**

The **Synthetic Data Generator Evaluative Visualization Tool** uses **Generative Adversarial Networks (GANs)** to create and evaluate synthetic data, crucial for privacy preservation and data balance in machine learning. This tool provides **real-time visualization** and **metrics** to give instant feedback on synthetic data quality.

Key features:
- **Iterative, user-guided generation** to minimize computational load, especially for large datasets.
- **Customizable parameters** such as learning rate, batch size, and more, to fine-tune the generation process.
- **Real-time metrics** like **losses** and **KL divergences** displayed on the left, allowing you to assess data quality at each iteration.
- **Dynamic visualizations** that show the proximity of synthetic data to original data, helping users decide when to stop the generation process.

## **How to Run Locally**

1. **Clone the repository**:

   ```
   git clone https://github.com/anonymsynthetic/SDGEV.git
   ```
   ```
   cd ./Synthetic-Data-Generator-Evaluative-Visualization-Tool
    ```
2. **Create and activate a virtual environment**:
   ```
   python3 -m venv sdenv
   ```
   ```
   source sdenv/bin/activate
    ```
3. **Install the required packages**:
   ```
   pip install -r requirements.txt
   ```
4. **Run the app**:
   ```
   ./sdenv/bin/python -m streamlit run src/main.py
   ```
   After running, open the provided local URL (e.g., http://localhost:8501) in your browser.

## **Tool Usage**


Key features:
- **Parameter selection**: Choose parameters like learning rate, batch size, etc., before starting the synthetic data generation.
- **Real-time feedback**: Once generation starts, the app displays performance metrics such as losses and KL divergences on the left panel.
- **Visualization**: The tool visualizes the similarity between synthetic and original data in real-time, helping you evaluate data quality and stop the process when desired.

For more details on the project, read the full research paper here:
https://github.com/Ahmetyasin/Synthetic-Data-Generator-Evaluative-Visualization-Tool/blob/main/docs/Paper.pdf


![Tool Demo](https://github.com/Ahmetyasin/Synthetic-Data-Generator-Evaluative-Visualization-Tool/blob/main/img/SDGEV_vid.gif)
  





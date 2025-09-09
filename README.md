# FIFA Player Market Value Prediction ⚽️💰

This project is an end-to-end machine learning application that predicts the market value of football players. The model is built using XGBoost and deployed as an interactive web application with Streamlit.

---

## 🚀 Live Demo

**[Check out the live application here!](https://fifaplayervalueprediction.streamlit.app/)**

---

## 🖼️ App Preview



This application allows you to input various player attributes, such as overall rating, potential, and age, to receive a real-time prediction of their market value in Euros.

---

## 🛠️ Tech Stack

-   **Language:** **Python**
-   **Data Analysis:** **Pandas**, **NumPy**
-   **Model Building:** **Scikit-learn**, **XGBoost**
-   **Web Framework:** **Streamlit**
-   **Development:** **Jupyter Notebook**

---

## 📂 Project Structure

A brief overview of the key files and directories in this repository.
├── app.py                  # Main Streamlit application script <br>
├── requirements.txt        # Python dependencies for deployment <br>
├── assets/                 # Contains images and other static assets <br>
├── data/                   # Contains raw and processed datasets <br>
├── models/                 # Contains the trained .pkl model file <br>
└── notebooks/              # Contains the Jupyter Notebook for analysis <br>

---

## ⚙️ How to Run Locally

To run this project on your own machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/DhruvJainPy/FifaPlayerValuePrediction.git](https://github.com/DhruvJainPy/FifaPlayerValuePrediction.git)
    cd FifaPlayerValuePrediction
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    # For Mac/Linux
    python3 -m venv env
    source env/bin/activate

    # For Windows
    python -m venv env
    .\env\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    Your web browser will automatically open to the application's local URL.

# 🎬 Movie Recommender System: An Interactive AI-Powered Web App  

Welcome to the **Movie Recommender System**!  
This project is an **end-to-end interactive web application** that provides **personalized movie recommendations** using **collaborative filtering** techniques. The app offers direct links to movie pages on **TMDb**.

---

## 🚀 Features  
✅ **Personalized Movie Recommendations**  
- Uses **Collaborative Filtering** with **k-Nearest Neighbors (kNN)** and **SVD** for dimensionality reduction.  

✅ **Interactive Web Application**  
- A fully responsive web interface built with **Flask** and **Bootstrap**, featuring clickable links to each movie’s **TMDb** pages.  

✅ **Modular, Scalable Codebase**  
- Clean **modular structure**, with **exception handling** and **logging** for a robust, scalable application.

---

## 🧱 Project Structure 
📂 src

┣ 📂 components

┃ ┣ 📄 data_ingestion.py # Loads and processes datasets

┃ ┣ 📄 data_transformation.py # Transforms data and computes utility matrices

┃ ┗ 📄 model_trainer.py # Builds recommendation models

┣ 📂 pipeline

┃ ┣ 📄 train_pipeline.py # End-to-end training pipeline

┃ ┗ 📄 predict_pipeline.py # Prediction pipeline for recommendations

┣ 📄 app.py # Flask web application

┣ 📄 exception.py # Exception handling

┣ 📄 logger.py # Logging setup

┗ 📄 utils.py # Utility functions


---

## 📊 Models & Algorithms Used  
- **Collaborative Filtering** using **k-Nearest Neighbors (kNN)**  
- **Dimensionality Reduction** using **Truncated SVD**  
- **Utility Matrix Construction** for user-item interactions  
- **Bayesian Average Ratings** to handle biased ratings  

---

## 🖥️ Technologies Used  
| **Technology** | **Purpose**                      |
|----------------|----------------------------------|
| Python         | Core programming language        |
| Flask          | Web framework for the app        |
| scikit-learn   | Machine learning algorithms      |
| pandas         | Data manipulation                |
| numpy          | Numerical operations             |
| Bootstrap      | Frontend design and UI styling   |
| fuzzywuzzy     | String matching for recommendations |

---

## 🎨 User Interface Preview  
✨ The web app features a **colorful and modern UI** with:  
- **Search functionality** to find movies  
- **Clickable links** to TMDb pages  
- **Recommendations** generated in real-time  

---

## 📂 Data Sources  
- **MovieLens Dataset**: `movies.csv` and `ratings.csv`  
- **Links Dataset**: `links.csv` (for IMDb/TMDb hyperlinks)  

---

## 🔧 How to Run the Project Locally  

1️⃣ **Clone the Repository**  

git clone https://github.com/your-username/movie-recommender-system.git  
cd movie-recommender-system  

2️⃣ Create a Virtual Environment & Install Dependencies

python -m venv venv  
source venv/bin/activate   # For Linux/Mac  
venv\Scripts\activate      # For Windows  
pip install -r requirements.txt  

3️⃣ Run the Web App

python app.py  

4️⃣ Access the Web App
Open your browser and visit:
👉 http://localhost:5000



# Phishing Website Detection Using Hybrid Machine Learning

## Project Overview
The phishing website detection system is a web-based application built using a hybrid ensemble of XGBoost and LightGBM. The system is designed to identify phishing websites and protect users from online fraud.

The system analyzes website URLs and determines whether a website is phishing or legitimate based on extracted URL-based features. This project was designed and developed by me as part of my MCA academic project (VTU) using Python, Flask, and machine learning techniques.

## Objectives
- Detect phishing websites using URL-based feature analysis
- Improve detection accuracy using a hybrid machine learning approach
- Reduce false positives and false negatives
- Provide real-time phishing detection through a web application
- Store and manage URL scan history

## Technologies Used
### Frontend
- HTML
- CSS

### Backend
- Python 3.8
- Flask

### Machine Learning
- XGBoost
- LightGBM
- Hybrid Ensemble Model

### Database
- SQLite

### Tools and Libraries
- NumPy
- Pandas
- Scikit-learn
- Git and GitHub

## Key Features
### User Module
- Enter website URLs for phishing detection
- View phishing or legitimate results
- Display prediction confidence
- View previously scanned URLs

### Admin Module
- Monitor scanned URLs
- View detection results
- Track system activity

## System Architecture
The system follows a client–server architecture. The frontend handles user interaction, while the backend performs URL preprocessing, feature extraction, and classification using the hybrid XGBoost and LightGBM model. The SQLite database is used to store scan history and prediction results.

## How to Run the Project
### Prerequisites
- Python 3.8 or above
- Flask
- Required Python libraries

### Steps
- Open the project folder
- Install required dependencies
- Run the Flask application
- Access the application through a web browser
- Enter a URL to check whether it is phishing or legitimate

## Conclusion
The phishing website detection system effectively identifies phishing websites using a hybrid ensemble of XGBoost and LightGBM. The system provides accurate results, supports real-time detection, and offers a simple web-based interface suitable for practical use.

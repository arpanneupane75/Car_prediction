# 🚗 Car Price Prediction Web App

A comprehensive **Car Price Prediction** application built with **Machine Learning** models and deployed with **Streamlit**. This project predicts the price of a car based on its features using various regression techniques, providing users a quick and interactive way to estimate vehicle prices.

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Model Performance & Findings](#model-performance--findings)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## About the Project

This project aims to build a robust predictive model for car prices using a dataset of various car attributes such as horsepower, engine size, fuel type, and more. It compares several regression models — including Lasso, Ridge, ElasticNet, SVR, KNN, and Polynomial Regression — to find the best-performing algorithm.

Users can interactively select important car features via a dynamic UI built with Streamlit and obtain a predicted price instantly.

---

## Features

- **Multiple regression models** available for prediction:
  - Support Vector Regression (SVR)
  - K-Nearest Neighbors (KNN)
  - Polynomial Regression
  - Lasso Regression
  - Ridge Regression
  - ElasticNet Regression

- **Dynamic UI** for feature input with:
  - Number inputs for numeric features with realistic value ranges
  - Radio buttons for categorical features with all available categories

- **One-hot encoding** and **scaling** consistent with training preprocessing for accurate predictions

- **Model details** and **coefficients** display for interpretability (for linear models)

- **Extensive metadata** management to ensure feature consistency and integrity

---

## Model Performance & Findings

| Model                 | R² Score | RMSE       | MAE        | Notes                                           |
|-----------------------|----------|------------|------------|------------------------------------------------|
| **Lasso Regression**  | 0.7953   | 4020.17    | 2565.42    | Best overall performance with good feature selection and regularization |
| **ElasticNet**        | 0.7944   | 4029.21    | 2521.47    | Similar performance to Lasso, balances L1 and L2 penalties |
| **Ridge Regression**  | 0.7919   | 4053.01    | 2519.07    | Comparable to Lasso, favors smaller coefficients |
| **SVR**               | 0.7801   | 4166.12    | 2590.29    | Non-linear model, slightly less performant here |
| **KNN Regression**    | 0.2896   | 7488.73    | 3981.95    | Underperformed, likely due to curse of dimensionality |
| **Polynomial Regression** | -0.0337  | 9033.54    | 5521.99    | Poor fit, possibly overfitting or not suitable |

### Important Features Impacting Price

Top features influencing price prediction (confirmed by model coefficients and domain knowledge):

- **Horsepower**
- **Engine Size**
- **Curb Weight**
- **City MPG & Highway MPG**
- **Car Body Style**
- **Fuel Type**
- **Drive Wheel Configuration**

These features have significant influence on car pricing and were prioritized for UI input to enhance prediction relevance.

---

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**

```bash
git clone https://github.com/arpanneupane75/Car_prediction.git
cd Car_prediction
```

2. **Create and activate a virtual environment (recommended)**

```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

```



3. **Install required dependencies**

```bash
pip install -r requirements.txt
```



4. **Run the Streamlit app**

```bash
streamlit run app.py
```
## Usage

*   Use the sidebar to select your preferred regression model.
*   Input car features via the numeric sliders and categorical radio buttons.
*   Click "Predict Price" to get the estimated car price.
*   View the input summary and model coefficients (if available for the selected model) in the expandable sections for better understanding.

## Project Structure
```bash
├── app.py # Streamlit app for prediction UI
├── CarPrice_Assignment.csv # Original dataset
├── model_training.ipynb # Jupyter notebook for model development and evaluation
├── scaler.joblib # Scaler saved for preprocessing
├── svr_model.joblib # Saved SVR model
├── knn_model.joblib # Saved KNN model
├── poly_model.joblib # Saved Polynomial Regression model
├── poly_features.joblib # Polynomial features transformer
├── lasso_model.joblib # Saved Lasso model
├── ridge_model.joblib # Saved Ridge model
├── elastic_model.joblib # Saved ElasticNet model
├── metadata.joblib # Metadata for feature columns and categories
├── requirements.txt # Required Python packages
└── README.md
```

## Future Enhancements

*   **Implement confidence intervals or prediction uncertainty:** Provide users with an understanding of the model's confidence in its predictions.
*   **Add support for more granular feature inputs:** Incorporate additional car attributes such as manufacturing year, specific brand models, or trim levels.
*   **Integrate external APIs for live data fetching and updates:** Allow the application to fetch real-time car data or update its dataset periodically.
*   **Provide visualization of feature importance dynamically:** Display interactive charts showing which features have the most significant impact on car prices for the selected model.
*   **Deploy on cloud platforms for wider accessibility:** Make the application publicly available by deploying it on services like Heroku, AWS, Google Cloud, or Azure.

---

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1.  Fork the repository.
2.  Create a new branch:
    ```bash
    git checkout -b feature/YourFeature
    ```
3.  Commit your changes:
    ```bash
    git commit -m 'Add YourFeature'
    ```
4.  Push to the branch:
    ```bash
    git push origin feature/YourFeature
    ```
5.  Create a Pull Request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file in the repository for more details.

---

## Contact

**Arpan Neupane**
*   **Email:** arpanneupane75@gmail.com
*   **GitHub:** [https://github.com/arpanneupane75](https://github.com/arpanneupane75)

Thank you for visiting this project! Feel free to try out different models and features for your car price prediction needs.


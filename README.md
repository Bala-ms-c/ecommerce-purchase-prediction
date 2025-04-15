# Ecommerce Purchase Prediction

This project is an implementation of a machine learning model to predict customer behavior on an ecommerce website. The model uses XGBoost to predict if a user will make a purchase (Revenue) based on various features such as page views, duration, and traffic type.

## Project Structure

ecommerce_purchase_prediction/ 

├── data/ # Data files 

├── models/ # Trained model and preprocessor 

├── src/ # Source code for data processing and model training 

├── dags/ # Airflow DAGs for automation 

├── prediction_results.png # Visualized model performance 

├── requirements.txt # List of required dependencies 

├── README.md # Project documentation


## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Bala-ms-c/ecommerce-purchase-prediction.git
    cd ecommerce-purchase-prediction
    ```

2. Create a virtual environment:

    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:

    - For **Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    - For **macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Model:

1. To run the XGBoost model and generate predictions:

    - Set up your Airflow DAG (`xgboost_prediction_and_chart`).
    - Execute the DAG to train the model and generate prediction charts.

2. To manually run the prediction script, you can use:

    ```bash
    python src/preprocessing_pipeline.py
    ```

### Airflow DAG:

The project includes an Airflow DAG to automate model predictions and visualize results. Make sure Airflow is properly configured on your local machine or server. The DAG will:

- Load the dataset.
- Apply preprocessing.
- Train the XGBoost model.
- Generate prediction results and a chart of model accuracy.

## Files

- **`xgboost_model.pkl`**: The trained XGBoost model.
- **`preprocessor.pkl`**: The preprocessing model.
- **`online_shoppers_intention.csv`**: Raw dataset for training.
- **`prediction_results.png`**: A visualization of model accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

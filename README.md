# IMDB Movie Review Sentiment Analysis

This project is a sentiment analysis tool built using a Recurrent Neural Network (RNN) model trained on the IMDB movie reviews dataset. The model predicts whether a given movie review expresses a **positive** or **negative** sentiment. It is deployed using **Streamlit**, allowing users to upload a CSV file containing movie reviews and get real-time sentiment predictions, which can then be downloaded as a CSV file.

## Key Features:
- **Pre-trained Sentiment Analysis Model**: The model is based on a Simple RNN architecture, trained using TensorFlow/Keras on the IMDB dataset.
- **Text Preprocessing**: Automatically handles text preprocessing, including tokenization and padding, to prepare the review for sentiment analysis.
- **Batch Processing**: Upload a CSV file with a column of movie reviews and get predictions for each review.
- **Downloadable Predictions**: The output CSV contains the original reviews and their corresponding sentiment predictions (positive or negative).
- **Streamlit Integration**: The web app allows a user-friendly interface for interacting with the model, with file upload and download functionality.

## How It Works:
1. **Upload a CSV file**: Users upload a CSV file containing a column named `'review'` with movie review text.
2. **Sentiment Prediction**: The pre-trained RNN model processes each review and predicts whether the sentiment is positive or negative.
3. **Download Results**: After processing, the user can download a CSV file with the original reviews and their predicted sentiment labels.

## Technologies Used:
- **TensorFlow**: For building and training the RNN model.
- **Keras**: High-level API for TensorFlow, used for the model architecture.
- **Streamlit**: To create the web interface for the app.
- **Pandas**: For handling CSV data.
- **NumPy**: For efficient numerical computations.

## Usage:
1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/imdb-sentiment-analysis.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
4. Interact with the model by uploading a CSV file containing movie reviews.


Link to STreamlit app https://imdb-sentiment-analysis-using-rnn-rnn.streamlit.app/



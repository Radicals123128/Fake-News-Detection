# Fake News Detection Web Application

This is a Flask-based web application that uses machine learning to detect fake news. The application provides a simple interface where users can input news text and get predictions about whether the news is likely to be real or fake.

## Features

- User-friendly web interface
- Real-time news analysis
- Machine learning-based prediction
- Confidence score for predictions

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your web browser and navigate to:
```
http://localhost:5000
```

## Project Structure

- `app.py`: Main Flask application file
- `templates/index.html`: Frontend HTML template
- `model/`: Directory for storing trained model files
- `requirements.txt`: List of Python dependencies

## Model Training

The application expects two pickle files in the `model` directory:
- `fake_news_model.pkl`: The trained classification model
- `vectorizer.pkl`: The fitted TF-IDF vectorizer

If these files are not present, the application will create simple placeholder models. For production use, you should train your own models with appropriate data.

## Technologies Used

- Flask
- scikit-learn
- pandas
- Bootstrap
- jQuery
- HTML/CSS/JavaScript
# Profile Analysis System

This system analyzes text input to extract various attributes and preferences, particularly useful for relationship/dating profile analysis. It uses NLP techniques with spaCy and includes machine learning capabilities for age prediction.

## Setup

1. Make sure you have Python 3.13 installed (this version is required for compatibility with the latest package versions).

2. Install the required dependencies:
```bash
# First, upgrade pip and install build tools
python -m pip install --upgrade pip
pip install wheel setuptools

# Install the dependencies
pip install -r requirements.txt

# Finally, download the spaCy English language model
python -m spacy download en_core_web_sm
```

## Required Model Files

The system requires the following model files to be present in the working directory:
- `best_rf_model.pkl` - Random Forest model for age prediction
- `scaler.pkl` - Scaler for feature normalization
- `encoders.pkl` - Label encoders for categorical features

## Usage

Run the script:
```bash
python keyword_extractor.py
```

Enter your text when prompted, and the system will return:
- Extracted attributes and preferences
- Categorized information including:
  - Age & Age Range
  - Height
  - Religion & Caste
  - Language & Ethnicity
  - Diet & Lifestyle
  - Smoking/Drinking Habits
  - Education & Profession
  - Values & Personality Traits
  - Location & Relocation Preferences
  - Hobbies & Interests
  - Relationship & Family Views
  - Spiritual/Religious Inclinations
  - Other Preferences

## Example Input
```
I am a 28-year-old vegetarian from India, 5'8" tall. I speak English and Hindi, and I'm interested in music and books. I am a non-smoker and work as a software engineer.
```

## Features
- Text preprocessing and normalization
- Pattern-based attribute extraction
- Named entity recognition using spaCy
- Categorical feature encoding
- Age prediction using machine learning
- Comprehensive category matching
- Handles various formats of age, height, and other attributes

## Troubleshooting

If you encounter any installation issues:

1. Make sure you have the latest pip:
```bash
python -m pip install --upgrade pip
```

2. Install build tools:
```bash
pip install wheel setuptools
```

3. Try installing dependencies one by one in this specific order:
```bash
pip install numpy>=1.26.0
pip install spacy>=3.7.2
pip install scikit-learn>=1.3.2
pip install pandas>=2.1.0
pip install joblib>=1.3.2
```

4. If you get model file errors, ensure all required .pkl files are in the working directory:
- best_rf_model.pkl
- scaler.pkl
- encoders.pkl

Note: If you're using an older version of Python (3.7-3.12), you'll need to use the older package versions from the previous requirements.txt file. 
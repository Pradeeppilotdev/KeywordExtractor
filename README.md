# Keyword Extraction

A sophisticated semantic preference tagging system that extracts and categorizes keywords from text input, particularly designed for relationship/dating profile analysis. The system uses advanced NLP techniques with spaCy and FastText word vectors for precise keyword extraction and semantic similarity matching.

## Features

- **Precise Keyword Extraction**: Extracts specific keywords and phrases from text using pattern matching and semantic analysis
- **Multi-Category Tagging**: Categorizes extracted keywords into 12 different preference categories
- **Semantic Similarity**: Uses FastText word vectors for intelligent keyword matching
- **CSV-Based Training Data**: Leverages comprehensive CSV datasets for accurate keyword recognition
- **Hybrid Extraction**: Combines exact matching and semantic similarity for optimal results

## Extracted Categories

The system extracts and categorizes keywords into the following categories:

1. **Religion & Caste** - Religious background and caste information
2. **Language & Ethnicity** - Languages spoken and ethnic background
3. **Diet & Lifestyle** - Dietary preferences and lifestyle choices
4. **Smoking / Drinking Habits** - Smoking and alcohol consumption patterns
5. **Education & Profession** - Educational background and professional information
6. **Values & Personality Traits** - Personal values and character traits
7. **Hobbies & Interests** - Recreational activities and interests
8. **Relationship & Family Views** - Views on relationships and family
9. **Spiritual/Religious Inclinations** - Spiritual practices and religious activities
10. **Height** - Physical height information
11. **Age & Age Range** - Age and age range preferences
12. **Location & Relocation Preferences** - Geographic preferences and location data

## Setup

1. **Python Requirements**: Python 3.7+ (recommended: Python 3.8+)

2. **Install Dependencies**:
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Download spaCy English language model
python -m spacy download en_core_web_sm
```

3. **Required Files**:
   - `keyword_extractor.py` - Main extraction script
   - `CSV/` directory containing training data files:
     - `religion_caste.csv`
     - `language.csv`
     - `diet_lifestyle.csv`
     - `education.csv`
     - `profession.csv`
     - `Values _ Personality Traits.csv`
     - `Hobbies_ Interests.csv`
     - `spiritual_religious.csv`
     - `location.csv`
     - And other category-specific CSV files

## Usage

Run the keyword extraction system:
```bash
python keyword_extractor.py
```

Enter your text when prompted. The system will analyze the input and return categorized keywords in JSON format.

## Example Input
```
I'm seeking a patient and emotionally resilient partner in her early 30s from a Gujarati-speaking Patel background, ideally around 5'2" tall and fluent in Gujarati and English. She follows a vegetarian diet rooted in family traditions, enjoys preparing regional sweets during festivals like Janmashtami and Diwali, and lives a lifestyle free from smoking and alcohol. She holds a degree in civil engineering and currently works as a sustainability consultant in Ahmedabad, focusing on green infrastructure in tier-2 cities.
```

## Example Output
```json
{
  "Religion & Caste": ["patel"],
  "Language & Ethnicity": ["English", "Gujarati"],
  "Values & Personality Traits": ["emotionally resilient"],
  "Diet & Lifestyle": ["follows vegetarian diet"],
  "Hobbies & Interests": ["preparing regional sweets"],
  "Education & Profession": ["civil engineering", "sustainability consultant"],
  "Height": ["5'2"],
  "Age & Age Range": ["early 30s"],
  "Location & Relocation Preferences": ["Ahmedabad"],
  "Smoking / Drinking habits": ["lifestyle free from smoking and alcohol"]
}
```

## Technical Details

### Extraction Methods
- **Pattern-Based Extraction**: Uses regex patterns for structured data like height, age, and location
- **Semantic Similarity**: Leverages FastText word vectors for fuzzy keyword matching
- **Exact Matching**: Precise matching for religion, caste, and other specific terms
- **Named Entity Recognition**: Uses spaCy NER for location and entity extraction

### Data Sources
- **CSV Training Data**: Comprehensive datasets for each category
- **Hardcoded Fallbacks**: Backup data for categories without CSV files
- **Dynamic Loading**: Automatically loads and processes CSV data at runtime

### Accuracy Features
- **Word Boundary Protection**: Prevents substring matches (e.g., "nat" in "coordinate")
- **Context-Aware Extraction**: Considers surrounding words and phrases
- **Duplicate Prevention**: Removes redundant and overlapping keywords
- **Category-Specific Logic**: Custom extraction rules for different categories

## Troubleshooting

### Common Issues

1. **CSV Loading Errors**: Ensure all CSV files are in the `CSV/` directory
2. **Model Download Issues**: Run `python -m spacy download en_core_web_sm` if spaCy model is missing
3. **Package Conflicts**: Use a virtual environment to avoid dependency conflicts

### Performance Tips
- The system loads FastText vectors on startup (may take a few seconds)
- Large text inputs are processed efficiently using optimized algorithms
- CSV data is cached after initial loading for faster subsequent runs

## Dependencies

- `spacy` - NLP processing and named entity recognition
- `gensim` - FastText word vectors
- `scikit-learn` - Cosine similarity calculations
- `pandas` - CSV data processing
- `numpy` - Numerical operations
- `re` - Regular expression processing


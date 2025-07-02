import spacy
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import re
import pandas as pd
import itertools

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load FastText vectors
print("Loading FastText vectors...")
word_vectors = api.load("fasttext-wiki-news-subwords-300")

# Load tag samples from CSVs
def load_tag_samples():
    tag_samples = {}
    
    try:
        # Religion & Caste - handle comma-separated format
        df_religion = pd.read_csv('CSV/religion_caste.csv', header=None, names=['phrase', 'category'], engine='python', on_bad_lines='skip')
        religion_phrases = set()
        for _, row in df_religion.iterrows():
            phrase_val = row['phrase']
            if isinstance(phrase_val, str) and phrase_val.strip():
                # Handle comma-separated phrases in a single cell
                phrases = phrase_val.split(',')
                for phrase in phrases:
                    phrase = phrase.strip()
                    if phrase and len(phrase) > 1:
                        religion_phrases.add(phrase)
        tag_samples['Religion & Caste'] = sorted(religion_phrases)
    except Exception as e:
        print(f"Error loading religion_caste.csv: {e}")
        tag_samples['Religion & Caste'] = []

    try:
        # Language & Ethnicity
        df_lang = pd.read_csv('CSV/language.csv')
        if 'Value' in df_lang.columns:
            tag_samples['Language & Ethnicity'] = sorted(set(df_lang['Value'].dropna().astype(str)))
        else:
            # Fallback to first column
            tag_samples['Language & Ethnicity'] = sorted(set(df_lang.iloc[:, 0].dropna().astype(str)))
    except Exception as e:
        print(f"Error loading language.csv: {e}")
        tag_samples['Language & Ethnicity'] = [
            "Kannada", "Tamil", "Telugu", "Malayalam", "Hindi", "Marathi", "Gujarati", 
            "Punjabi", "Bengali", "Odia", "Assamese", "Sanskrit", "Urdu", "English", 
            "French", "German", "Spanish", "Japanese", "Korean", "Chinese", "Mandarin", 
            "Russian", "Arabic", "Hebrew", "Pali", "Prakrit", "Tulu", "Konkani", "Sindhi", 
            "Nepali", "Sinhalese", "Bodo", "Santhali", "Maithili", "Dogri", "Manipuri", 
            "Kashmiri", "Bhili", "Gondi", "Khasi", "Mizo", "Bodo", "Santali",
            "South Indian", "North Indian", "Gujarati", "Punjabi", "Bengali", "Marwari", 
            "Sindhi", "Malayali", "Goan", "Konkani", "Rajasthani", "Bihari", "Assamese", 
            "Manipuri", "Nepali", "Tibetan", "Adivasi", "Tribal", "Anglo Indian", 
            "Dravidian", "Aryan", "Indo-Aryan", "Indo-European", "Indo-Dravidian"
        ]

    try:
        # Values & Personality Traits
        df_values = pd.read_csv('CSV/Values _ Personality Traits.csv', header=None)
        tag_samples['Values & Personality Traits'] = sorted(set(df_values[0].dropna().astype(str)))
    except Exception as e:
        print(f"Error loading Values _ Personality Traits.csv: {e}")
        tag_samples['Values & Personality Traits'] = [
            "charismatic", "entrepreneurial", "open-minded", "traditional", "dynamic", 
            "clear-headed", "progressive", "family values", "compassion", "shared growth", 
            "mutual respect", "adventurous", "warm", "thoughtful", "mindful living", 
            "cultural values", "inner growth", "spiritual", "funny", "humorous", "serious", 
            "calm", "principled", "ambitious", "hardworking", "dedicated", "loyal", 
            "honest", "trustworthy", "caring", "supportive", "understanding", "patient", 
            "optimistic", "pessimistic", "realistic", "practical", "idealistic", "introvert", 
            "extrovert", "ambivert", "assertive", "confident", "shy", "reserved", "outgoing", 
            "friendly", "sociable", "empathetic", "generous", "kind", "respectful", 
            "disciplined", "organized", "punctual", "responsible", "reliable", "independent", 
            "team player"
        ]

    try:
        # Spiritual/Religious Inclinations
        df_spirit = pd.read_csv('CSV/spiritual_religious.csv')
        if 'phrase' in df_spirit.columns:
            tag_samples['Spiritual/Religious Inclinations'] = sorted(set(df_spirit['phrase'].dropna().astype(str)))
        else:
            tag_samples['Spiritual/Religious Inclinations'] = sorted(set(df_spirit.iloc[:, 0].dropna().astype(str)))
    except Exception as e:
        print(f"Error loading spiritual_religious.csv: {e}")
        tag_samples['Spiritual/Religious Inclinations'] = [
            "spiritual", "religious", "meditation", "prayer", "temple", "church", "mosque", 
            "synagogue", "faith", "rituals", "spiritually inclined", "pilgrimage", "fasting", 
            "yoga", "chanting", "scripture study", "devotional music", "bhajans", "kirtan", 
            "mass", "namaz", "puja", "festivals", "temple festivals", "Ugadi celebrations", 
            "Ramadan", "Lent", "Diwali", "Christmas", "Eid", "Hanukkah", "Passover", 
            "Buddha Purnima", "Guru Nanak Jayanti"
        ]

    try:
        # Diet & Lifestyle - handle two-column format
        df_diet = pd.read_csv('CSV/diet_lifestyle.csv', header=None, names=['phrase', 'category'])
        diet_phrases = set()
        for _, row in df_diet.iterrows():
            phrase_val = row['phrase']
            if isinstance(phrase_val, str) and phrase_val.strip():
                diet_phrases.add(phrase_val.strip())
        tag_samples['Diet & Lifestyle'] = sorted(diet_phrases)
    except Exception as e:
        print(f"Error loading diet_lifestyle.csv: {e}")
        tag_samples['Diet & Lifestyle'] = [
            "vegetarian", "non-vegetarian", "vegan", "eggetarian", "lacto-vegetarian", 
            "ovo-vegetarian", "strict vegetarian", "pure vegetarian", "non-smoker", "smoker", 
            "teetotaler", "drinks occasionally", "does not drink", "alcoholic", "healthy lifestyle", 
            "balanced diet", "minimalist lifestyle", "active lifestyle", "fitness enthusiast", 
            "yoga practitioner", "gym goer", "sports enthusiast", "outdoor lover"
        ]

    try:
        # Hobbies & Interests
        df_hobbies = pd.read_csv('CSV/Hobbies_ Interests.csv', header=None)
        tag_samples['Hobbies & Interests'] = sorted(set(df_hobbies[0].dropna().astype(str)))
    except Exception as e:
        print(f"Error loading Hobbies_ Interests.csv: {e}")
        tag_samples['Hobbies & Interests'] = [
            "music", "books", "reading", "travel", "sports", "dancing", "painting", 
            "nature trails", "community service", "cultural exploration", "indie music", 
            "coffee", "meaningful conversations", "gardening", "pottery", "long walks", 
            "sunrise", "photography", "poetry", "art", "heritage sarees", "solo travel", 
            "sci-fi films", "Telugu literature", "Tamil identity", "mentoring", "artisans", 
            "public health", "public health coordinator", "folk music", "kolam art", 
            "weekend drives", "cooking", "baking", "yoga", "meditation", "hiking", 
            "trekking", "cycling", "swimming", "running", "jogging", "gym", "fitness", 
            "volunteering", "blogging", "vlogging", "podcasting", "board games", 
            "video games", "chess", "puzzles", "filmmaking", "acting", "theatre", 
            "bird watching", "astrophotography", "star gazing", "fashion", "makeup", 
            "styling", "interior design", "DIY", "crafts", "knitting", "embroidery", 
            "calligraphy", "languages", "learning languages", "debating", "public speaking", 
            "storytelling"
        ]

    try:
        # Education & Profession
        df_prof = pd.read_csv('CSV/profession.csv')
        df_edu = pd.read_csv('CSV/education.csv', header=None)
        prof_phrases = set()
        if 'phrase' in df_prof.columns:
            prof_phrases.update(df_prof['phrase'].dropna().astype(str))
        else:
            prof_phrases.update(df_prof.iloc[:, 0].dropna().astype(str))
        edu_phrases = set(df_edu[0].dropna().astype(str))
        tag_samples['Education & Profession'] = sorted(prof_phrases | edu_phrases)
    except Exception as e:
        print(f"Error loading profession.csv or education.csv: {e}")
        tag_samples['Education & Profession'] = [
            "engineer", "software engineer", "data scientist", "civil servant", "IAS", 
            "IPS", "IFS", "doctor", "MBBS", "dentist", "CA", "chartered accountant", 
            "professor", "lecturer", "teacher", "principal", "business owner", "entrepreneur", 
            "startup founder", "business analyst", "public health coordinator", "school teacher", 
            "fashion technology", "fashion boutique", "design merchandising", "business", 
            "M.Tech", "B.Tech", "B.E.", "M.E.", "B.Sc", "M.Sc", "B.Com", "M.Com", "B.A.", 
            "M.A.", "PhD", "postgraduate", "graduate", "degree", "arts", "humanities", 
            "commerce", "science", "lawyer", "advocate", "legal professional", "banker", 
            "investment banker", "consultant", "management consultant", "project manager", 
            "product manager", "marketing manager", "sales manager", "operations manager", 
            "HR manager", "accountant", "auditor", "architect", "civil engineer", 
            "mechanical engineer", "electrical engineer", "electronics engineer", 
            "IT professional", "software developer", "web developer", "UI/UX designer", 
            "graphic designer", "artist", "musician", "writer", "journalist", "editor", 
            "photographer", "filmmaker", "actor", "dancer", "sports coach", "athlete", 
            "researcher", "scientist", "biologist", "chemist", "physicist", "mathematician", 
            "statistician", "economist", "psychologist", "counselor", "therapist", 
            "social worker", "NGO worker", "government employee", "private sector", 
            "public sector", "retired", "homemaker"
        ]

    # Add missing tags that don't have CSV files
    tag_samples['Smoking / Drinking habits'] = [
        "smoker", "non-smoker", "drinker", "non-drinker", "teetotaler", "drinks occasionally", 
        "does not drink", "alcoholic", "occasional smoker", "social drinker", "never smoked", 
        "never drinks", "quit smoking", "quit drinking"
    ]
    
    tag_samples['Relationship & Family Views'] = [
        "family oriented", "values family", "joint family", "nuclear family", "close to parents", 
        "independent", "open to long distance", "wants children", "does not want children", 
        "open to adoption", "prefers pets", "traditional family values", "modern family values", 
        "open to inter-caste marriage", "open to inter-religious marriage", "prefers arranged marriage", 
        "prefers love marriage", "open to live-in relationship", "prefers early marriage", 
        "prefers late marriage", "single parent", "divorced", "widowed", "separated"
    ]

    # Load location data from CSV
    try:
        df_location = pd.read_csv('CSV/location.csv')
        if 'Location' in df_location.columns:
            tag_samples['Location & Relocation Preferences'] = sorted(set(df_location['Location'].dropna().astype(str)))
        else:
            # Fallback to first column
            tag_samples['Location & Relocation Preferences'] = sorted(set(df_location.iloc[:, 0].dropna().astype(str)))
    except Exception as e:
        print(f"Error loading location.csv: {e}")
        # Fallback to hardcoded list
        tag_samples['Location & Relocation Preferences'] = [
            "Kochi", "Thrissur", "Thiruvananthapuram", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Mumbai", "Delhi", "Gurgaon", "Noida", "Kolkata", "Madurai", "Tirunelveli", "Vizag", "Coimbatore", "Trichy", "Salem", "Tiruchengode", "Kochi", "Trivandrum", "Calicut", "Mangalore", "Udupi", "Hubli", "Belgaum", "Ahmedabad", "Surat", "Vadodara", "Jaipur", "Lucknow", "Patna", "Bhopal", "Indore", "Nagpur", "Raipur", "Ranchi", "Bhubaneswar", "Guwahati", "Shillong", "Imphal", "Aizawl", "Agartala", "Sikkim"
        ]

    return tag_samples

# Load tag_samples from CSVs
print("Loading tag samples from CSVs...")
tag_samples = load_tag_samples()

# Compute tag anchors
def phrase_vector(phrase):
    words = [w for w in phrase.lower().split() if w in word_vectors]
    if not words:
        return None
    return np.mean(np.array([word_vectors[w] for w in words]), axis=0)

tag_anchors = {}
for tag, samples in tag_samples.items():
    vecs = [phrase_vector(s) for s in samples]
    vecs = [v for v in vecs if v is not None]
    if vecs:
        tag_anchors[tag] = np.mean(np.array(vecs), axis=0)

# Targeted phrase extraction for exact keywords
def extract_phrases(text):
    doc = nlp(text)
    phrases = set()
    
    # Exact noun chunks and entities (1-3 words)
    phrases.update(chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3)
    phrases.update(ent.text for ent in doc.ents if len(ent.text.split()) <= 3)
    
    # Adjective + noun and compound phrases
    for token in doc:
        if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
            phrase = f"{token.text} {token.head.text}"
            if len(phrase.split()) <= 3:
                phrases.add(phrase)
        if token.dep_ == "compound":
            phrase = f"{token.text} {token.head.text}"
            if len(phrase.split()) <= 3:
                phrases.add(phrase)
    
    # Limited n-grams (1-3 words)
    words = [token.text for token in doc if not token.is_punct and not token.is_stop]
    for n in range(1, 4):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n])
            if len(phrase.split()) <= 3 and phrase_vector(phrase) is not None:
                phrases.add(phrase)
    
    # Filter low-quality phrases
    return [p for p in phrases if len(p) > 2 and len(p.split()) <= 3]

# Hybrid: precise extraction for Height, Age, Location
def extract_height_phrases(text):
    # Support both straight and curly quotes for feet/inches
    patterns = [
        r"\b\d{1,2}['’`]\d{1,2}(?:[\"”])?\b",   # 5'3, 5'3, 5'3", 5'3"
        r"\b\d{3}\s*cm\b",        # 180 cm
        r"\b\d\s*feet\b"          # 6 feet
    ]
    return re.findall("|".join(patterns), text)

def extract_age_phrases(text):
    patterns = [
        r"early\s*\d{2}s\b",   # early 30s
        r"mid\s*\d{2}s\b",     # mid 20s
        r"late\s*\d{2}s\b",    # late 40s
        r"\b\d{2}\s*(?:years?|yrs?)\s*old\b",  # 32 years old
        r"age\s*\d{2}\b"      # age 32
    ]
    return re.findall("|".join(patterns), text)

def extract_location_phrases(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC") and len(ent.text.split()) <= 2]

def extract_language_ethnicity_phrases(text):
    found = set()
    for lang in tag_samples["Language & Ethnicity"]:
        pattern = r"\b" + re.escape(lang) + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            found.add(lang)
    return list(found)

# Hybrid tagging: exact matching for Religion & Caste, semantic for others
def semantic_tagging(phrases, tag_anchors, text, base_threshold=0.8):
    results = {tag: [] for tag in tag_anchors}
    
    # Exact matching for Religion & Caste
    for phrase in phrases:
        if phrase.lower() in [s.lower() for s in tag_samples["Religion & Caste"]]:
            results["Religion & Caste"].append(phrase)
            continue
        
        # Semantic similarity for other tags
        vec = phrase_vector(phrase)
        if vec is None:
            continue
        best_tag, best_sim = None, 0
        for tag, anchor in tag_anchors.items():
            if tag == "Religion & Caste":  # Skip semantic for Religion & Caste
                continue
            sim = cosine_similarity(vec.reshape(1, -1), anchor.reshape(1, -1))[0][0]
            threshold = base_threshold + 0.05 * (3 - len(phrase.split()))  # Favor short phrases
            if sim > threshold and sim > best_sim:
                best_tag, best_sim = tag, sim
        if best_tag:
            results[best_tag].append(phrase)
    
    # Post-processing: cleanup and deconfliction
    for tag in results:
        # Remove redundant phrases (keep most specific)
        results[tag] = list(set(results[tag]))
        results[tag] = sorted(results[tag], key=lambda x: (-len(x.split()), x))[:3]  # Keep top 3 by specificity
        
        # Reassign misclassified phrases
        for phrase in results[tag][:]:
            if tag == "Religion & Caste" and phrase.lower() in ["tamil identity", "telugu literature", "brand tamil"]:
                results[tag].remove(phrase)
                results["Hobbies & Interests"].append(phrase)
            if tag == "Hobbies & Interests" and "mentoring" in phrase.lower():
                results[tag].remove(phrase)
                results["Education & Profession"].append(phrase)
    
    # Remove empty tags
    return {k: v for k, v in results.items() if v}

# Add lists of known Indian cities and states for better location extraction
INDIAN_CITIES = [
    "Kochi", "Thrissur", "Thiruvananthapuram", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Mumbai", "Delhi", "Gurgaon", "Noida", "Kolkata", "Madurai", "Tirunelveli", "Vizag", "Coimbatore", "Trichy", "Salem", "Tiruchengode", "Kochi", "Trivandrum", "Calicut", "Mangalore", "Udupi", "Hubli", "Belgaum", "Ahmedabad", "Surat", "Vadodara", "Jaipur", "Lucknow", "Patna", "Bhopal", "Indore", "Nagpur", "Raipur", "Ranchi", "Bhubaneswar", "Guwahati", "Shillong", "Imphal", "Aizawl", "Agartala", "Sikkim"
]
INDIAN_STATES = [
    "Kerala", "Tamil Nadu", "Karnataka", "Andhra Pradesh", "Telangana", "Maharashtra", "Gujarat", "Punjab", "West Bengal", "Uttar Pradesh", "Rajasthan", "Haryana", "Bihar", "Madhya Pradesh", "Odisha", "Assam", "Jharkhand", "Chhattisgarh", "Goa", "Tripura", "Meghalaya", "Manipur", "Nagaland", "Arunachal Pradesh", "Mizoram", "Sikkim"
]
COUNTRIES = ["India", "USA", "UK", "Europe", "Australia", "Canada", "Singapore", "Malaysia", "Middle East", "Dubai", "Abu Dhabi", "Qatar", "Saudi Arabia", "Japan", "Korea"]



# --- Improved Extraction for All Tags ---
import re

def extract_religion_caste_phrases(text, tag_list):
    found = set()
    text_lower = text.lower()
    # Whole word/phrase match (use word boundaries for all tags)
    for tag in tag_list:
        # Only match as a whole word or phrase, not as a substring
        # Use word boundaries and ensure it's not part of a larger word
        pattern = rf'\b{re.escape(tag)}\b'
        if re.search(pattern, text_lower):
            found.add(tag)
    # Parenthetical/compound patterns
    patterns = [
        r'([A-Za-z ]+)\s*\(([^)]+)\) background',
        r'([A-Za-z ]+)\s*\(([^)]+)\) community',
        r'([A-Za-z ]+)\s*\(([^)]+)\) family',
        r'([A-Za-z ]+)\s*\(([^)]+)\) caste'
    ]
    for pat in patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            for group in match.groups():
                candidate = group.strip().lower()
                for tag in tag_list:
                    if candidate == tag or tag in candidate:
                        found.add(tag)
    # Contextual patterns
    context_patterns = [r'([A-Za-z ]+) background', r'([A-Za-z ]+) community', r'([A-Za-z ]+) family', r'([A-Za-z ]+) caste']
    for pat in context_patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            candidate = match.group(1).strip().lower()
            for tag in tag_list:
                if candidate == tag or tag in candidate:
                    found.add(tag)
    # Prefer most specific (longest) match, suppress generic if specific exists
    if found:
        max_len = max(len(x.split()) for x in found)
        specific = {x for x in found if len(x.split()) == max_len}
        # Remove generic tags if a more specific one exists
        generics = {"hindu", "muslim", "christian", "sikh", "buddhist", "jain", "parsi", "jewish"}
        if any(x not in generics for x in specific):
            specific = {x for x in specific if x not in generics}
        return sorted(specific)
    return []

def extract_hobbies_phrases(text, tag_list):
    found = set()
    # Pattern-based
    patterns = [r'enjoys ([^.,;]+)', r'loves ([^.,;]+)', r'in her free time,? ([^.,;]+)', r'spends time ([^.,;]+)', r'likes ([^.,;]+)', r'fond of ([^.,;]+)', r'passionate about ([^.,;]+)']
    for pat in patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            phrase = match.group(1).strip()
            if phrase:
                found.add(phrase)
    # Dictionary-based (fuzzy/partial)
    text_lower = text.lower()
    for tag in tag_list:
        if tag in text_lower:
            found.add(tag)
    # Prefer most specific
    if found:
        max_len = max(len(x.split()) for x in found)
        found = {x for x in found if len(x.split()) == max_len}
    return sorted(found)

# For other tags, you can apply similar logic:
def extract_tag_phrases(text, tag_list, patterns=None):
    found = set()
    text_lower = text.lower()
    if patterns:
        for pat in patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                phrase = match.group(1).strip()
                if phrase:
                    found.add(phrase)
    for tag in tag_list:
        if re.search(rf'\b{re.escape(tag)}\b', text_lower):
            found.add(tag)
    if found:
        max_len = max(len(x.split()) for x in found)
        found = {x for x in found if len(x.split()) == max_len}
    return sorted(found)

# Improved Smoking / Drinking habits extraction
# This will catch phrases like 'free from smoking and alcohol', 'lifestyle free from smoking', etc.
def extract_smoking_drinking_phrases(text, tag_list):
    found = set()
    text_lower = text.lower()
    # Pattern-based extraction
    patterns = [
        r'free from smoking( and alcohol)?',
        r'lifestyle free from smoking( and alcohol)?',
        r'free of smoking( and alcohol)?',
        r'avoids smoking( and alcohol)?',
        r'does not smoke',
        r'does not drink',
        r'non-smoker',
        r'non-drinker',
        r'teetotaler',
        r'drinks occasionally',
        r'alcoholic',
        r'smoker',
        r'smoking alcohol completely',
        r'no smoking',
        r'no alcohol',
        r'quit smoking',
        r'quit drinking'
    ]
    for pat in patterns:
        for match in re.finditer(pat, text_lower):
            phrase = match.group(0).strip()
            if phrase:
                found.add(phrase)
    # Dictionary-based (whole word/phrase)
    for tag in tag_list:
        pattern = rf'(?<![\w]){re.escape(tag)}(?![\w])'
        if re.search(pattern, text_lower):
            found.add(tag)
    if found:
        max_len = max(len(x.split()) for x in found)
        found = {x for x in found if len(x.split()) == max_len}
    return sorted(found)

# Improved location extraction using CSV data
def extract_location_phrases_improved(text, tag_list):
    found = set()
    text_lower = text.lower()
    
    # Filter out non-location terms that might be in the CSV
    non_location_terms = {
        "diwali", "janmashtami", "ugadi", "bonalu", "thaipusam", "navaratri", 
        "ramadan", "eid", "christmas", "hanukkah", "passover", "seva", "vratas",
        "vratams", "puja", "bhajans", "kirtan", "mass", "namaz", "fasting",
        "pilgrimage", "temple", "church", "mosque", "synagogue", "festivals",
        "celebrations", "rituals", "devotion", "spiritual", "religious"
    }
    
    # Use spaCy NER for GPE/LOC
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            for part in re.split(r",|/| or | and |\\bor\\b|\\band\\b", ent.text):
                part = part.strip()
                if part and part.lower() not in non_location_terms:
                    found.add(part)
    
    # Dictionary-based matching from CSV (whole word/phrase)
    for tag in tag_list:
        if tag.lower() not in non_location_terms:
            pattern = rf'\b{re.escape(tag)}\b'
            if re.search(pattern, text, re.IGNORECASE):
                found.add(tag)
    
    return sorted(found)

# Update main extraction/tagging flow
if __name__ == "__main__":
    print("=== Semantic Preference Tagger [Precise Keywords] ===")
    while True:
        print("Type your preferences below (or type 'exit' to quit):")
        user_input = input().strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        phrases = extract_phrases(user_input)
        tagged = semantic_tagging(phrases, tag_anchors, user_input, base_threshold=0.8)
        
        # Hybrid: override/add precise tags
        height_phrases = extract_height_phrases(user_input)
        if height_phrases:
            tagged["Height"] = height_phrases
        age_phrases = extract_age_phrases(user_input)
        if age_phrases:
            tagged["Age & Age Range"] = age_phrases
        # Improved location extraction using CSV data
        location_phrases = extract_location_phrases_improved(user_input, tag_samples['Location & Relocation Preferences'])
        if location_phrases:
            tagged["Location & Relocation Preferences"] = location_phrases
        language_ethnicity_phrases = extract_language_ethnicity_phrases(user_input)
        if language_ethnicity_phrases:
            tagged["Language & Ethnicity"] = language_ethnicity_phrases
        # Improved smoking/drinking habits
        smoking_drinking_phrases = extract_smoking_drinking_phrases(user_input, tag_samples['Smoking / Drinking habits'])
        if smoking_drinking_phrases:
            tagged["Smoking / Drinking habits"] = smoking_drinking_phrases
        # Remove smoking/drinking phrases from Diet & Lifestyle if present
        if "Diet & Lifestyle" in tagged and "Smoking / Drinking habits" in tagged:
            for phrase in tagged["Smoking / Drinking habits"]:
                if phrase in tagged["Diet & Lifestyle"]:
                    tagged["Diet & Lifestyle"].remove(phrase)
        # Improved hobbies/interests
        hobbies_phrases = extract_hobbies_phrases(user_input, tag_samples['Hobbies & Interests'])
        if hobbies_phrases:
            tagged["Hobbies & Interests"] = hobbies_phrases
        # Improved education/profession
        education_profession_phrases = extract_tag_phrases(user_input, tag_samples['Education & Profession'])
        if education_profession_phrases:
            tagged["Education & Profession"] = education_profession_phrases
        # Improved values/personality
        values_personality_phrases = extract_tag_phrases(user_input, tag_samples['Values & Personality Traits'])
        if values_personality_phrases:
            tagged["Values & Personality Traits"] = values_personality_phrases
        # Improved relationship/family views
        relationship_family_phrases = extract_tag_phrases(user_input, tag_samples['Relationship & Family Views'])
        if relationship_family_phrases:
            tagged["Relationship & Family Views"] = relationship_family_phrases
        # Improved spiritual/religious
        spiritual_phrases = extract_tag_phrases(user_input, tag_samples['Spiritual/Religious Inclinations'])
        if spiritual_phrases:
            tagged["Spiritual/Religious Inclinations"] = spiritual_phrases
        # Improved religion & caste extraction
        religion_caste_phrases = extract_religion_caste_phrases(user_input, tag_samples['Religion & Caste'])
        if religion_caste_phrases:
            tagged["Religion & Caste"] = religion_caste_phrases
        
        # Always output all required tags, even if empty
        required_tags = [
            "Religion & Caste",
            "Language & Ethnicity",
            "Diet & Lifestyle",
            "Smoking / Drinking habits",
            "Education & Profession",
            "Values & Personality Traits",
            "Hobbies & Interests",
            "Relationship & Family Views",
            "Spiritual/Religious Inclinations"
        ]
        for tag in required_tags:
            if tag not in tagged:
                tagged[tag] = []
        
        print("\n=== Extracted Tags ===")
        print(json.dumps(tagged, indent=2, ensure_ascii=False))
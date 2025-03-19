from FictionalClassification import levenshtein, fuzzy_match, double_metaphone_match, is_dictionary_word
import joblib
import pandas as pd

# Load model and references
model = joblib.load('fictional_name_classifier.pkl')
reference_real = joblib.load('reference_real.pkl')
reference_fictional = joblib.load('reference_fictional.pkl')
reference_real_metaphone = joblib.load('reference_real_metaphone.pkl')
reference_fictional_metaphone = joblib.load('reference_fictional_metaphone.pkl')


def predict_fictionality(name):
    lev_real = levenshtein(name, reference_real)
    lev_fictional = levenshtein(name, reference_fictional)
    fuzzy_real = fuzzy_match(name, reference_real)
    fuzzy_fictional = fuzzy_match(name, reference_fictional)
    doublemetaphone_real = double_metaphone_match(name, reference_real_metaphone)
    doublemetaphone_fict = double_metaphone_match(name, reference_fictional_metaphone)
    dict_word = is_dictionary_word(name)

    features = pd.DataFrame([[lev_real, lev_fictional, fuzzy_real, fuzzy_fictional, doublemetaphone_real,
                              doublemetaphone_fict, dict_word]],
                            columns=['levenshtein_real', 'levenshtein_fictional', 'fuzzy_real', 'fuzzy_fictional',
                                     'double_metaphone_real', 'double_metaphone_fictional', 'is_dictionary_word'])

    # Get Probabilities
    prob_fictional = model.predict_proba(features)[0][1]

    # Set a threshold
    threshold = 0.8
    prediction = 1 if prob_fictional >= threshold else 0

    # Predict with RandomForestClassifier
    print("\n📌 **Name Analysis:**", name)
    print(f"🔹 Suspicious Probability:            {prob_fictional}")
    print(f"🔹 Levenshtein Distance (Real):       {lev_real}")
    print(f"🔹 Levenshtein Distance (Fictional):  {lev_fictional}")
    print(f"🔹 Fuzzy Matching (Real):             {fuzzy_real}")
    print(f"🔹 Fuzzy Matching (Fictional):        {fuzzy_fictional}")
    print(f"🔹 DMetaphone Match (Real):           {'✅ Match' if doublemetaphone_real else '❌ No Match'}")
    print(f"🔹 DMetaphone Match (Fictional):      {'✅ Match' if doublemetaphone_fict else '❌ No Match'}")
    print(f"🔹 Contains a Dictionnary Word:       {dict_word}")
    return 'Suspicious' if prediction == 1 else 'Not Suspicious'


# Example Prediction
names = [
    "John Wick",
    "SpiderMan Smith",
    "Wolf Heimer",
    "Knee Ellen",
    "Mickey Mouse",
    "Mickey Mousse",
    "Harry Potter",
    "Ice Queen",
    "Tao Tao",
    "Annie Ngo",
    "Bat man",
    "Boot man",
    "Apple Juice",
    "Donald Trump",
    "Christina Perez",
    "Alexandre Gagnon",
    "Jack Sparrow",
    "Eric Brault",
    "Kane Yu-Kis Mi",
    "Ai Wan Tyu",
    "Youssef Hamza",
    "Sonia Creo",
    "Mary Wang",
    "Vladimir Kosov",
    "Vladimir Putin",
    "Affan Pazheri",
    "Yuen Tao Wang",
    "John Doe",
    "John Cena",
    "Tiger Woods",
    "Nelson Mandela",
    "Kobe Bryant",
    "Ching Chong"
]

for name in names:
    print(predict_fictionality(name))



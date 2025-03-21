from FictionalClassification import levenshtein, fuzzy_match, double_metaphone_match, is_proper_noun
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
    proper_noun = is_proper_noun(name)

    features = pd.DataFrame([[lev_real, lev_fictional, fuzzy_real, fuzzy_fictional, doublemetaphone_real,
                              doublemetaphone_fict, proper_noun]],
                            columns=['levenshtein_real', 'levenshtein_fictional', 'fuzzy_real', 'fuzzy_fictional',
                                     'double_metaphone_real', 'double_metaphone_fictional', 'is_proper_noun'])

    # Get Probabilities
    prob_fictional = model.predict_proba(features)[0][1]

    # Set a threshold
    threshold = 0.9
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
    print(f"🔹 Contains a Proper Noun:            {proper_noun}")
    return 'Suspicious' if prediction == 1 else 'Not Suspicious'


# Example Prediction
global_names = [
    # Vietnamese Names
    "Nguyen Van A", "Tran Thi Bich", "Le Hoang Nam", "Pham Minh Anh", "Hoang Thanh Tung",
    "Doan Hai Dang", "Bui Ngoc Han", "Vo Quoc Bao", "Dang Van Dung", "Ta Thi Mai",

    # Chinese Names
    "Wang Wei", "Zhang Li", "Chen Jie", "Liu Yang", "Huang Mei",
    "Zhao Min", "Xu Feng", "Sun Lei", "Deng Rong", "Gao Yan",

    # Korean Names
    "Kim Min-Jae", "Lee Ji-Eun", "Park Ji-Hoon", "Choi Soo-Young", "Jung Hye-Jin",
    "Kang Dong-Won", "Yoon Seo-Jin", "Shin Hye-Sung", "Song Joon-Ki", "Ryu Hwa-Young",

    # Indian Names
    "Amit Sharma", "Priya Kapoor", "Rajesh Kumar", "Neha Patel", "Suresh Reddy",
    "Deepika Nair", "Ananya Iyer", "Ravi Srinivasan", "Meera Choudhary", "Tarun Gupta",

    # Arabic Names
    "Mohammed Al-Farsi", "Aisha Khalid", "Omar Nasser", "Layla Hussein", "Yusuf Rahman",
    "Fatima Al-Mansoori", "Khalid Ibrahim", "Zainab Ahmed", "Hassan Karim", "Samira Hafez",

    # Russian Names
    "Dmitry Ivanov", "Natalia Petrova", "Sergey Smirnov", "Ekaterina Sokolova", "Alexei Orlov",
    "Anna Pavlova", "Viktor Kuznetsov", "Olga Morozova", "Nikolai Fedorov", "Elena Volkova",

    # French Names
    "Jean Dupont", "Camille Bernard", "Louis Moreau", "Isabelle Laurent", "Pierre Lefevre",
    "Sophie Dubois", "Antoine Rousseau", "Juliette Girard", "Mathieu Fournier", "Clara Chevalier",

    # Spanish Names
    "Carlos Fernández", "Isabella Ramirez", "José López", "Ana Sánchez", "Francisco Torres",
    "Lucia Moreno", "Miguel Herrera", "Carmen Castillo", "Santiago Vargas", "Maria Gonzalez",

    # German Names
    "Johannes Müller", "Klara Schmidt", "Lukas Fischer", "Hannah Weber", "Felix Wagner",
    "Sophia Hoffmann", "Sebastian Becker", "Lea Schulz", "Niklas Richter", "Mia Schneider",

    # Italian Names
    "Giovanni Rossi", "Francesca Ricci", "Luca Conti", "Giulia Bianchi", "Matteo Greco",
    "Elena Romano", "Alessandro Ferrara", "Valentina Rizzo", "Davide Moretti", "Aurora Gallo",

    # African Names
    "Kwame Mensah", "Fatou Diop", "Tendai Chikore", "Adebola Okafor", "Lerato Mthembu",
    "Abdoulaye Ndiaye", "Zuberi Juma", "Ayaan Mohamed", "Nia Kamau", "Moussa Diallo",
]


weird_names = [
    "Toilet Seat", "Expired Milk", "404 Not Found", "XXx_EliteSniper_xX", "Unnamed User",
    "Password123", "John Doe", "No Signal", "Baby Shark", "Captain Obvious",
    "Error Message", "First Name Last Name", "Null Void", "Just Some Guy",
    "Burger McFries", "Banana Hammock", "Wifi Password", "Ima Robot",
    "Duck Quackington", "Doctor Strange Love", "Lord Voldemort Jr.",
    "Extra Chromosome", "Yolo Swaggins", "Sir Lancelot of Camelot",
    "Gucci FlipFlop", "Lil Test Tube", "McLovin", "Coconut Head",
    "Dank Memer", "Zeus Almighty", "Sir Fartsalot", "Google Translate",
    "Emoji Overlord", "Captain Planet", "Chairman Meow", "Sith Lord Karen"
]


for name in global_names:
    print(predict_fictionality(name))



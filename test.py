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
    threshold = 0.85
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
legit_names = [
    "James Smith", "Maria Garcia", "Robert Johnson", "Emily Brown", "John Williams",
    "David Jones", "Michael Miller", "William Davis", "Joseph Martinez", "Charles Anderson",
    "Daniel Lee", "Matthew Taylor", "Anthony Thomas", "Christopher Harris", "Andrew Clark",
    "Joshua Lewis", "Alexander Walker", "Sophia Young", "Olivia Hall", "Benjamin Allen",
    "Isabella Hernandez", "Mason King", "Ethan Wright", "Liam Scott", "Emma Green",
    "Lucas Adams", "Charlotte Baker", "Henry Nelson", "Amelia Carter", "Sebastian Mitchell",
    "Ava Perez", "Samuel Roberts", "Harper Turner", "Jack Phillips", "Evelyn Campbell",
    "Daniel Rodriguez", "Chloe Parker", "Nathan Moore", "Madison Evans", "Jonathan Edwards",
    "Grace Collins", "David Stewart", "Zoey Rivera", "Christopher Sanchez", "Julian Flores",
    "Elijah Gomez", "Victoria Butler", "Noah Murphy", "Hannah Bryant", "Caleb Cooper",
    "Scarlett Hughes", "Isaac Russell", "Lillian Griffin", "Gabriel Peterson", "Sophie Fisher",
    "Nathaniel Powell", "Brooklyn Kim", "Aaron Sanders", "Leah Morris", "Owen Watson",
    "Eleanor Price", "Hunter Torres", "Violet Richardson", "Eli Wood", "Penelope Brooks",
    "Xavier Bennett", "Aria Gray", "Adam Barnes", "Natalie Long", "Miles Foster",
    "Savannah Bailey", "Christian Jenkins", "Addison Perry", "Dominic Howard", "Zoe Bell",
    "Leo Ross", "Paisley Morgan", "Hudson Scott", "Delilah Bailey", "Carson Reed",
    "Aurora Cooper", "Elias Ward", "Gabriella Rogers", "Everett Cox", "Camila Adams",
    "Micah Simmons", "Lucy Butler", "Landon Price", "Autumn Powell", "Roman Stewart",
    "Clara Bryant", "Jameson Nguyen", "Juliette Myers", "Easton Bennett", "Hazel Foster",
    "Weston Bailey", "Piper Sanders", "Maxwell Jenkins", "Adeline Ramirez", "Beckett Torres",
    "Lydia Gray", "Zane Reed", "Sienna Sullivan", 'Mary Wang', 'Affan Pazheri', 'Laure Colins',
    'Laure Collins', 'Yuen PAO Wang', 'Justin Trudeau', 'Jimmy Carter', 'Berny Sanderz'
]

non_legit_names = [
    # Famous Fictional Characters
    "Bruce Wayne", "Clark Kent", "Peter Parker", "Tony Stark", "Walter White",
    "Sherlock Holmes", "Frodo Baggins", "Darth Vader", "Luke Skywalker", "Indiana Jones",
    "James Bond", "Homer Simpson", "Bart Simpson", "Marge Simpson", "Rick Sanchez",
    "Morty Smith", "Bugs Bunny", "Daffy Duck", "Scooby Doo", "Shrek",
    "Donkey", "SpongeBob SquarePants", "Mickey Mouse", "Jack Sparrow", "Harry Potter",

    # Meme / Joke Names
    "Joe Mama", "Hugh Jass", "Ben Dover", "Mike Hunt", "Al Beback",
    "Anita Bath", "Seymour Butts", "Rick O’Shea", "Pat Myback", "Bob Loblaw",
    "Ima Pigg", "Ivana Tinkle", "Moe Lester", "Dixie Normous", "Major Wood",

    # Fantasy / Sci-Fi Inspired Names
    "Zyler Vex", "Ronan Drakos", "Kai Zenth", "Nova Quinn", "Ezra Volante",
    "Xander Nyx", "Seraphina Lux", "Orin Vale", "Lyric Noir", "Artemis Riven",
    "Caspian Thorn", "Selene Nightshade", "Zayden Crowe", "Astra Bellamy", "Vesper Lin",

    # Slightly Realistic but Still Fictional Names
    "Elliot Graves", "Derek Caldwell", "Julian Mercer", "Amelia Vaughn",
    "Silas Montgomery", "Rosalind Faulkner", "Lillian Hawthorne", "Maximillian Stokes",
    "Eleanor Sterling", "Damian Everly", "Celeste Holloway", "Vincent Langley",
    "Isla Whitmore", "Nicolette Fontaine", "Theodore Winslow",

    # Completely Random / Absurd Names
    "Accounts Payable", "Ching Chong", "Bomboclat", "I love you",
    "M. Taylor", "Velkan Starforge", "Zenthrax Bloodfang", "Arkanis Shadowmere",
    "Drax Silvermoon", "Nyx Everdusk", "Vexxion Nightfall", "Zephyr Windwhisper",
    "Talon Emberfang", "Orion Moonshadow", "Kaelith Starborn", "Axel Ragnarok",
    "Mordecai Hellscream", "Shadowfang Evergloom", "Eryndor Blackthorn", "Krynn Valeria"
]

for name in non_legit_names:
    print(predict_fictionality(name))



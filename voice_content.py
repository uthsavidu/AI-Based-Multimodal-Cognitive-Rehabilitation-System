"""Multilingual content and language helpers for the Voice Cognitive Game.

Supports English, Tamil, Kannada and Hindi. Each language has its own:
  - Level 1 word-recall vocabulary
  - Level 2 sentence-recall sentences
  - Level 3 rapid category-naming categories, with translated display labels
    and a language-specific vocabulary used to validate spoken words

Category *keys* (e.g. "animals", "fruits") are shared, language-neutral
identifiers used internally (and sent between client/server); only the
displayed label and the accepted vocabulary differ per language. This keeps
scoring logic in voice_analysis.py language-aware without needing to touch
its comparison algorithms.

No ML/translation model is used here -- these are plain, curated,
deterministic word lists (same design choice as the original English-only
content), extended across languages and with more entries per language.
"""

from __future__ import annotations

from random import choice

SUPPORTED_LANGUAGES = {
    "en": "English",
    "ta": "Tamil",
    "kn": "Kannada",
    "hi": "Hindi",
}

# BCP-47 locale codes used by the browser's SpeechRecognition / speechSynthesis APIs.
SPEECH_LOCALE = {
    "en": "en-US",
    "ta": "ta-IN",
    "kn": "kn-IN",
    "hi": "hi-IN",
}

DEFAULT_LANGUAGE = "en"


def normalize_language(value: str | None) -> str:
    code = (value or "").strip().lower()
    return code if code in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def speech_locale(language: str | None) -> str:
    return SPEECH_LOCALE.get(normalize_language(language), SPEECH_LOCALE[DEFAULT_LANGUAGE])


# ---------------------------------------------------------------------------
# Level 1: single-word recall vocabulary
# ---------------------------------------------------------------------------

LEVEL1_WORDS: dict[str, list[str]] = {
    "en": [
        "apple", "chair", "river", "pencil", "garden", "window", "candle", "bridge", "orange", "bottle",
        "school", "flower", "doctor", "train", "cloud", "basket", "spoon", "market", "monkey", "rainbow",
        "book", "phone", "mountain", "button", "house", "table", "chicken", "elephant", "umbrella", "pillow",
        "mirror", "ladder", "hammer", "kitchen", "bicycle", "airplane", "forest", "island", "ocean", "desert",
    ],
    "ta": [
        "ஆப்பிள்", "நாற்காலி", "ஆறு", "பென்சில்", "தோட்டம்", "ஜன்னல்", "மெழுகுவர்த்தி", "பாலம்", "ஆரஞ்சு", "பாட்டில்",
        "பள்ளி", "பூ", "மருத்துவர்", "ரயில்", "மேகம்", "கூடை", "கரண்டி", "சந்தை", "குரங்கு", "வானவில்",
        "புத்தகம்", "தொலைபேசி", "மலை", "பொத்தான்", "வீடு", "மேசை", "கோழி", "யானை", "குடை", "தலையணை",
        "கண்ணாடி", "ஏணி", "சுத்தி", "சமையலறை", "மிதிவண்டி", "விமானம்", "காடு", "தீவு", "கடல்", "பாலைவனம்",
    ],
    "kn": [
        "ಸೇಬು", "ಕುರ್ಚಿ", "ನದಿ", "ಪೆನ್ಸಿಲ್", "ತೋಟ", "ಕಿಟಕಿ", "ಮೇಣದಬತ್ತಿ", "ಸೇತುವೆ", "ಕಿತ್ತಳೆ", "ಬಾಟಲಿ",
        "ಶಾಲೆ", "ಹೂವು", "ವೈದ್ಯ", "ರೈಲು", "ಮೋಡ", "ಬುಟ್ಟಿ", "ಚಮಚ", "ಮಾರುಕಟ್ಟೆ", "ಕೋತಿ", "ಕಾಮನಬಿಲ್ಲು",
        "ಪುಸ್ತಕ", "ಫೋನ್", "ಪರ್ವತ", "ಗುಂಡಿ", "ಮನೆ", "ಮೇಜು", "ಕೋಳಿ", "ಆನೆ", "ಕೊಡೆ", "ದಿಂಬು",
        "ಕನ್ನಡಿ", "ಏಣಿ", "ಸುತ್ತಿಗೆ", "ಅಡುಗೆಮನೆ", "ಸೈಕಲ್", "ವಿಮಾನ", "ಕಾಡು", "ದ್ವೀಪ", "ಸಮುದ್ರ", "ಮರುಭೂಮಿ",
    ],
    "hi": [
        "सेब", "कुर्सी", "नदी", "पेंसिल", "बगीचा", "खिड़की", "मोमबत्ती", "पुल", "संतरा", "बोतल",
        "स्कूल", "फूल", "डॉक्टर", "ट्रेन", "बादल", "टोकरी", "चम्मच", "बाज़ार", "बंदर", "इंद्रधनुष",
        "किताब", "फोन", "पहाड़", "बटन", "घर", "मेज़", "मुर्गी", "हाथी", "छाता", "तकिया",
        "आईना", "सीढ़ी", "हथौड़ा", "रसोई", "साइकिल", "हवाई जहाज़", "जंगल", "द्वीप", "समुद्र", "रेगिस्तान",
    ],
}


# ---------------------------------------------------------------------------
# Level 2: sentence-recall sentences
# ---------------------------------------------------------------------------

LEVEL2_SENTENCES: dict[str, list[str]] = {
    "en": [
        "The boy is playing with the ball.",
        "She walked slowly to the market.",
        "The dog barked at the mail carrier.",
        "He forgot his umbrella at the office.",
        "The children are reading a story book.",
        "A small bird sat beside the garden gate.",
        "My sister packed lunch before leaving home.",
        "The teacher asked everyone to read aloud.",
        "We planted flowers near the front window.",
        "The bus arrived shortly after the rain stopped.",
        "Grandfather told us a story after dinner.",
        "Please put the blue notebook on the table.",
        "The young girl carefully crossed the quiet road.",
        "A warm breeze moved the leaves in the park.",
        "They visited the library to borrow new books.",
        "The old clock on the wall stopped ticking.",
        "Our neighbor waters the plants every morning.",
        "The baby smiled when the music started playing.",
        "Rain fell gently on the quiet village roofs.",
        "The farmer carried fresh vegetables to the market.",
    ],
    "ta": [
        "சிறுவன் பந்துடன் விளையாடுகிறான்.",
        "அவள் மெதுவாக சந்தைக்கு நடந்தாள்.",
        "நாய் தபால்காரரைப் பார்த்து குரைத்தது.",
        "அவர் தனது குடையை அலுவலகத்தில் மறந்துவிட்டார்.",
        "குழந்தைகள் ஒரு கதைப் புத்தகத்தைப் படிக்கிறார்கள்.",
        "ஒரு சிறிய பறவை தோட்ட வாசலில் அமர்ந்தது.",
        "என் சகோதரி வீட்டை விட்டு கிளம்பும் முன் மதிய உணவை தயார் செய்தாள்.",
        "ஆசிரியர் அனைவரையும் சத்தமாக படிக்கச் சொன்னார்.",
        "நாங்கள் முன் ஜன்னலுக்கு அருகில் பூக்களை நட்டோம்.",
        "மழை நின்ற சிறிது நேரத்திலேயே பேருந்து வந்தது.",
        "தாத்தா இரவு உணவுக்குப் பிறகு எங்களுக்கு ஒரு கதை சொன்னார்.",
        "தயவுசெய்து நீல நோட்புக்கை மேசையில் வையுங்கள்.",
        "இளம் பெண் கவனமாக அமைதியான சாலையைக் கடந்தாள்.",
        "சூடான காற்று பூங்காவில் இலைகளை அசைத்தது.",
        "அவர்கள் புதிய புத்தகங்களை இரவல் வாங்க நூலகத்திற்குச் சென்றனர்.",
        "பழைய சுவர் கடிகாரம் ஓடுவதை நிறுத்தியது.",
        "எங்கள் அண்டை வீட்டார் தினமும் காலையில் செடிகளுக்கு தண்ணீர் ஊற்றுகிறார்.",
        "இசை தொடங்கியதும் குழந்தை புன்னகைத்தது.",
        "அமைதியான கிராம கூரைகளில் மழை மெதுவாகப் பெய்தது.",
        "விவசாயி புதிய காய்கறிகளை சந்தைக்குக் கொண்டு சென்றார்.",
    ],
    "kn": [
        "ಹುಡುಗ ಚೆಂಡಿನೊಂದಿಗೆ ಆಡುತ್ತಿದ್ದಾನೆ.",
        "ಅವಳು ನಿಧಾನವಾಗಿ ಮಾರುಕಟ್ಟೆಗೆ ನಡೆದಳು.",
        "ನಾಯಿ ಅಂಚೆಯಣ್ಣನ ಮೇಲೆ ಬೊಗಳಿತು.",
        "ಅವನು ತನ್ನ ಕೊಡೆಯನ್ನು ಕಚೇರಿಯಲ್ಲಿ ಮರೆತನು.",
        "ಮಕ್ಕಳು ಒಂದು ಕಥೆ ಪುಸ್ತಕವನ್ನು ಓದುತ್ತಿದ್ದಾರೆ.",
        "ಒಂದು ಪುಟ್ಟ ಹಕ್ಕಿ ತೋಟದ ಬಾಗಿಲ ಬಳಿ ಕುಳಿತಿತು.",
        "ನನ್ನ ಅಕ್ಕ ಮನೆ ಬಿಡುವ ಮೊದಲು ಊಟ ಕಟ್ಟಿದಳು.",
        "ಶಿಕ್ಷಕರು ಎಲ್ಲರಿಗೂ ಜೋರಾಗಿ ಓದಲು ಹೇಳಿದರು.",
        "ನಾವು ಮುಂಭಾಗದ ಕಿಟಕಿಯ ಬಳಿ ಹೂವುಗಳನ್ನು ನೆಟ್ಟಿದ್ದೇವೆ.",
        "ಮಳೆ ನಿಂತ ಸ್ವಲ್ಪ ಸಮಯದಲ್ಲೇ ಬಸ್ ಬಂದಿತು.",
        "ಅಜ್ಜ ಊಟದ ನಂತರ ನಮಗೆ ಒಂದು ಕಥೆ ಹೇಳಿದರು.",
        "ದಯವಿಟ್ಟು ನೀಲಿ ನೋಟ್‌ಬುಕ್ ಅನ್ನು ಮೇಜಿನ ಮೇಲೆ ಇಡಿ.",
        "ಚಿಕ್ಕ ಹುಡುಗಿ ಎಚ್ಚರಿಕೆಯಿಂದ ಶಾಂತ ರಸ್ತೆಯನ್ನು ದಾಟಿದಳು.",
        "ಬೆಚ್ಚಗಿನ ಗಾಳಿ ಉದ್ಯಾನದಲ್ಲಿ ಎಲೆಗಳನ್ನು ಅಲುಗಾಡಿಸಿತು.",
        "ಅವರು ಹೊಸ ಪುಸ್ತಕಗಳನ್ನು ಎರವಲು ಪಡೆಯಲು ಗ್ರಂಥಾಲಯಕ್ಕೆ ಹೋದರು.",
        "ಗೋಡೆಯ ಹಳೆಯ ಗಡಿಯಾರ ಓಡುವುದನ್ನು ನಿಲ್ಲಿಸಿತು.",
        "ನಮ್ಮ ನೆರೆಯವರು ಪ್ರತಿ ಬೆಳಿಗ್ಗೆ ಗಿಡಗಳಿಗೆ ನೀರು ಹಾಕುತ್ತಾರೆ.",
        "ಸಂಗೀತ ಪ್ರಾರಂಭವಾದಾಗ ಮಗು ನಕ್ಕಿತು.",
        "ಶಾಂತ ಹಳ್ಳಿಯ ಮಾಡುಗಳ ಮೇಲೆ ಮಳೆ ಮೆಲ್ಲಗೆ ಬಿದ್ದಿತು.",
        "ರೈತ ತಾಜಾ ತರಕಾರಿಗಳನ್ನು ಮಾರುಕಟ್ಟೆಗೆ ಹೊತ್ತೊಯ್ದನು.",
    ],
    "hi": [
        "लड़का गेंद के साथ खेल रहा है।",
        "वह धीरे-धीरे बाज़ार की ओर चली।",
        "कुत्ता डाकिए पर भौंका।",
        "वह अपना छाता दफ्तर में भूल गया।",
        "बच्चे एक कहानी की किताब पढ़ रहे हैं।",
        "एक छोटी चिड़िया बगीचे के फाटक के पास बैठी।",
        "मेरी बहन ने घर से निकलने से पहले खाना पैक किया।",
        "शिक्षक ने सबको ज़ोर से पढ़ने को कहा।",
        "हमने सामने वाली खिड़की के पास फूल लगाए।",
        "बारिश रुकने के थोड़ी देर बाद बस आई।",
        "दादाजी ने रात के खाने के बाद हमें एक कहानी सुनाई।",
        "कृपया नीली नोटबुक मेज़ पर रखें।",
        "छोटी लड़की ने सावधानी से शांत सड़क पार की।",
        "गर्म हवा ने पार्क में पत्तों को हिलाया।",
        "वे नई किताबें उधार लेने पुस्तकालय गए।",
        "दीवार पर पुरानी घड़ी चलना बंद हो गई।",
        "हमारा पड़ोसी हर सुबह पौधों को पानी देता है।",
        "संगीत शुरू होते ही बच्चा मुस्कुराया।",
        "शांत गाँव की छतों पर बारिश धीरे से गिरी।",
        "किसान ताज़ी सब्ज़ियाँ बाज़ार ले गया।",
    ],
}


# ---------------------------------------------------------------------------
# Level 3: category word-generation -- category keys, translated labels, and
# a per-language vocabulary used to validate spoken words.
# ---------------------------------------------------------------------------

# English keeps the full original category set, now with every category
# having a real vocabulary (the original build only had word lists for the
# first three, so the other nine always scored zero valid words).
LEVEL3_CATEGORY_LABELS_EN = {
    "animals": "animals",
    "fruits": "fruits",
    "vegetables": "vegetables",
    "household items": "household items",
    "jobs and professions": "jobs and professions",
    "transportation": "modes of transportation",
    "clothing": "clothing items",
    "things found at school": "things found at school",
    "body parts": "body parts",
    "sports": "sports",
    "flowers": "flowers",
    "things in a kitchen": "things found in a kitchen",
}

# Tamil, Kannada and Hindi offer a focused set of 8 categories -- each with
# a real, curated vocabulary in that language's script (adding more
# categories without matching vocabulary would unfairly score zero every
# time, so breadth is intentionally matched to actual coverage).
LEVEL3_CATEGORY_LABELS_TA = {
    "animals": "விலங்குகள்",
    "fruits": "பழங்கள்",
    "vegetables": "காய்கறிகள்",
    "household items": "வீட்டு பொருட்கள்",
    "body parts": "உடல் உறுப்புகள்",
    "clothing": "ஆடைகள்",
    "transportation": "போக்குவரத்து வகைகள்",
    "flowers": "பூக்கள்",
}

LEVEL3_CATEGORY_LABELS_KN = {
    "animals": "ಪ್ರಾಣಿಗಳು",
    "fruits": "ಹಣ್ಣುಗಳು",
    "vegetables": "ತರಕಾರಿಗಳು",
    "household items": "ಮನೆ ವಸ್ತುಗಳು",
    "body parts": "ದೇಹದ ಭಾಗಗಳು",
    "clothing": "ಬಟ್ಟೆಗಳು",
    "transportation": "ಸಾರಿಗೆ ವಿಧಗಳು",
    "flowers": "ಹೂವುಗಳು",
}

LEVEL3_CATEGORY_LABELS_HI = {
    "animals": "जानवर",
    "fruits": "फल",
    "vegetables": "सब्ज़ियाँ",
    "household items": "घरेलू सामान",
    "body parts": "शरीर के अंग",
    "clothing": "कपड़े",
    "transportation": "परिवहन के साधन",
    "flowers": "फूल",
}

LEVEL3_CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "en": LEVEL3_CATEGORY_LABELS_EN,
    "ta": LEVEL3_CATEGORY_LABELS_TA,
    "kn": LEVEL3_CATEGORY_LABELS_KN,
    "hi": LEVEL3_CATEGORY_LABELS_HI,
}

CATEGORY_WORD_LISTS: dict[str, dict[str, set[str]]] = {
    "en": {
        "animals": {
            "dog", "cat", "tiger", "lion", "horse", "cow", "goat", "sheep", "pig", "elephant",
            "giraffe", "zebra", "monkey", "bear", "wolf", "fox", "deer", "rabbit", "squirrel",
            "mouse", "rat", "bat", "kangaroo", "koala", "panda", "camel", "donkey", "buffalo",
            "hippo", "hippopotamus", "rhino", "rhinoceros", "cheetah", "leopard", "jaguar",
            "gorilla", "chimpanzee", "otter", "beaver", "raccoon", "hedgehog", "seal", "walrus",
            "dolphin", "whale", "shark", "octopus", "crab", "lobster", "shrimp", "jellyfish",
            "eagle", "hawk", "owl", "parrot", "sparrow", "pigeon", "crow", "peacock", "penguin",
            "ostrich", "flamingo", "duck", "goose", "swan", "chicken", "turkey", "rooster",
            "snake", "lizard", "crocodile", "alligator", "turtle", "tortoise", "frog", "toad",
            "spider", "ant", "bee", "wasp", "butterfly", "moth", "beetle", "grasshopper",
            "dragonfly", "ladybug", "scorpion", "snail", "worm",
        },
        "fruits": {
            "apple", "banana", "orange", "grape", "mango", "pineapple", "strawberry", "blueberry",
            "raspberry", "blackberry", "watermelon", "melon", "papaya", "guava", "kiwi", "peach",
            "plum", "pear", "cherry", "apricot", "lemon", "lime", "coconut", "pomegranate", "fig",
            "date", "grapefruit", "avocado", "lychee", "jackfruit", "passionfruit", "nectarine",
            "tangerine", "cantaloupe", "cranberry", "currant",
        },
        "vegetables": {
            "carrot", "potato", "tomato", "onion", "garlic", "cabbage", "cauliflower", "broccoli",
            "spinach", "lettuce", "cucumber", "pepper", "capsicum", "chili", "eggplant", "brinjal",
            "pumpkin", "squash", "zucchini", "beet", "beetroot", "radish", "turnip", "corn",
            "peas", "beans", "okra", "ginger", "mushroom", "celery", "asparagus", "kale",
            "artichoke", "leek",
        },
        "household items": {
            "chair", "table", "bed", "sofa", "lamp", "mirror", "blanket", "pillow", "broom",
            "bucket", "clock", "curtain", "carpet", "shelf", "cupboard", "stool", "fan",
            "mattress", "basket", "ladder", "mop", "dustbin", "vase", "candle", "cushion",
            "hanger", "rug", "almirah", "trunk", "kettle",
        },
        "jobs and professions": {
            "doctor", "teacher", "engineer", "farmer", "nurse", "police", "lawyer", "pilot",
            "driver", "chef", "carpenter", "electrician", "plumber", "painter", "tailor",
            "barber", "fisherman", "soldier", "scientist", "dentist", "clerk", "librarian",
            "mechanic", "gardener", "potter", "weaver", "shopkeeper", "journalist",
            "photographer", "accountant",
        },
        "transportation": {
            "car", "bus", "train", "bicycle", "motorcycle", "airplane", "boat", "ship", "truck",
            "taxi", "scooter", "tractor", "van", "rickshaw", "helicopter", "subway", "tram",
            "ferry", "cart", "skateboard",
        },
        "clothing": {
            "shirt", "pant", "saree", "skirt", "dress", "jacket", "sweater", "scarf", "hat",
            "cap", "gloves", "socks", "shoes", "sandals", "belt", "tie", "blouse", "trousers",
            "kurta", "dhoti", "coat", "shawl", "turban", "vest", "shorts",
        },
        "things found at school": {
            "book", "pencil", "pen", "eraser", "ruler", "bag", "desk", "chalk", "board",
            "notebook", "chair", "uniform", "classroom", "teacher", "playground", "library",
            "crayon", "sharpener", "globe", "clock", "scissors", "glue", "calculator", "map",
            "timetable",
        },
        "body parts": {
            "head", "eye", "ear", "nose", "mouth", "hand", "leg", "finger", "foot", "shoulder",
            "elbow", "knee", "neck", "chest", "back", "hair", "chin", "cheek", "tongue", "teeth",
            "eyebrow", "wrist", "ankle", "heel", "stomach",
        },
        "sports": {
            "cricket", "football", "hockey", "tennis", "badminton", "basketball", "volleyball",
            "swimming", "running", "cycling", "boxing", "wrestling", "chess", "kabaddi", "golf",
            "baseball", "rugby", "skating", "archery", "gymnastics",
        },
        "flowers": {
            "rose", "lily", "lotus", "jasmine", "marigold", "sunflower", "tulip", "daisy",
            "orchid", "hibiscus", "daffodil", "carnation", "lavender", "chrysanthemum",
            "magnolia", "dahlia", "poppy", "iris", "violet", "peony",
        },
        "things in a kitchen": {
            "spoon", "plate", "cup", "pan", "pot", "knife", "fork", "bowl", "stove", "oven",
            "kettle", "glass", "jar", "ladle", "tray", "sieve", "blender", "fridge", "sink",
            "grater", "whisk", "tongs",
        },
    },
    "ta": {
        "animals": {
            "நாய்", "பூனை", "புலி", "சிங்கம்", "குதிரை", "பசு", "ஆடு", "செம்மறியாடு", "பன்றி", "யானை",
            "ஒட்டகச்சிவிங்கி", "வரிக்குதிரை", "குரங்கு", "கரடி", "ஓநாய்", "நரி", "மான்", "முயல்", "அணில்",
            "எலி", "வவ்வால்", "கங்காரு", "பாண்டா", "ஒட்டகம்", "கழுதை", "எருமை", "காண்டாமிருகம்", "சிறுத்தை",
            "கோரில்லா", "ஆமை", "தவளை", "பாம்பு", "முதலை", "கழுகு", "ஆந்தை", "கிளி", "குருவி", "மயில்",
            "வாத்து", "கோழி",
        },
        "fruits": {
            "ஆப்பிள்", "வாழைப்பழம்", "ஆரஞ்சு", "திராட்சை", "மாம்பழம்", "அன்னாசி", "ஸ்ட்ராபெரி", "தர்பூசணி",
            "பப்பாளி", "கொய்யா", "கிவி", "பீச்", "பேரிக்காய்", "செர்ரி", "எலுமிச்சை", "தேங்காய்", "மாதுளை",
            "அத்திப்பழம்", "பலாப்பழம்", "இலந்தை",
        },
        "vegetables": {
            "கேரட்", "உருளைக்கிழங்கு", "தக்காளி", "வெங்காயம்", "பூண்டு", "முட்டைக்கோஸ்", "காலிஃபிளவர்",
            "ப்ரோக்கோலி", "கீரை", "வெள்ளரிக்காய்", "மிளகாய்", "கத்தரிக்காய்", "பூசணிக்காய்", "முள்ளங்கி",
            "சோளம்", "பட்டாணி", "பீன்ஸ்", "வெண்டைக்காய்", "இஞ்சி", "காளான்",
        },
        "household items": {
            "நாற்காலி", "மேசை", "கட்டில்", "விளக்கு", "கண்ணாடி", "போர்வை", "தலையணை", "துடைப்பம்", "வாளி",
            "கடிகாரம்", "திரைச்சீலை", "தரைவிரிப்பு", "அலமாரி", "மின்விசிறி", "கூடை", "ஏணி", "குப்பைத்தொட்டி",
            "குவளை", "மெழுகுவர்த்தி", "தொங்குதடி",
        },
        "body parts": {
            "தலை", "கண்", "காது", "மூக்கு", "வாய்", "கை", "கால்", "விரல்", "பாதம்", "தோள்", "முழங்கை",
            "முழங்கால்", "கழுத்து", "மார்பு", "முதுகு", "முடி", "கன்னம்", "நாக்கு", "பல்", "வயிறு",
        },
        "clothing": {
            "சட்டை", "பேண்ட்", "புடவை", "பாவாடை", "ஜாக்கெட்", "ஸ்வெட்டர்", "தொப்பி", "கையுறை", "காலுறை",
            "காலணி", "செருப்பு", "பெல்ட்", "டை", "கோட்", "சால்வை", "வேட்டி", "லுங்கி", "ஸ்கார்ஃப்", "பனியன்",
        },
        "transportation": {
            "கார்", "பேருந்து", "ரயில்", "மிதிவண்டி", "மோட்டார் சைக்கிள்", "விமானம்", "படகு", "கப்பல்",
            "லாரி", "டாக்ஸி", "ஸ்கூட்டர்", "டிராக்டர்", "ஆட்டோ", "ஹெலிகாப்டர்", "வேன்",
        },
        "flowers": {
            "ரோஜா", "லில்லி", "தாமரை", "மல்லிகை", "சாமந்தி", "சூரியகாந்தி", "டூலிப்", "ஆர்க்கிட்",
            "செம்பருத்தி", "மகரந்தம்",
        },
    },
    "kn": {
        "animals": {
            "ನಾಯಿ", "ಬೆಕ್ಕು", "ಹುಲಿ", "ಸಿಂಹ", "ಕುದುರೆ", "ಹಸು", "ಮೇಕೆ", "ಕುರಿ", "ಹಂದಿ", "ಆನೆ",
            "ಜಿರಾಫೆ", "ಜೀಬ್ರಾ", "ಮಂಗ", "ಕರಡಿ", "ತೋಳ", "ನರಿ", "ಜಿಂಕೆ", "ಮೊಲ", "ಅಳಿಲು", "ಇಲಿ",
            "ಬಾವಲಿ", "ಕಾಂಗರೂ", "ಪಾಂಡಾ", "ಒಂಟೆ", "ಕತ್ತೆ", "ಎಮ್ಮೆ", "ಘೇಂಡಾಮೃಗ", "ಚಿರತೆ", "ಆಮೆ", "ಕಪ್ಪೆ",
            "ಹಾವು", "ಮೊಸಳೆ", "ಹದ್ದು", "ಗೂಬೆ", "ಗಿಣಿ", "ಗುಬ್ಬಚ್ಚಿ", "ನವಿಲು", "ಬಾತುಕೋಳಿ", "ಕೋಳಿ",
        },
        "fruits": {
            "ಸೇಬು", "ಬಾಳೆಹಣ್ಣು", "ಕಿತ್ತಳೆ", "ದ್ರಾಕ್ಷಿ", "ಮಾವು", "ಅನಾನಸ್", "ಸ್ಟ್ರಾಬೆರಿ", "ಕಲ್ಲಂಗಡಿ",
            "ಪಪ್ಪಾಯಿ", "ಪೇರಲ", "ಕಿವಿ", "ಪೀಚ್", "ಪೇರ್", "ಚೆರ್ರಿ", "ನಿಂಬೆ", "ತೆಂಗಿನಕಾಯಿ", "ದಾಳಿಂಬೆ",
            "ಅಂಜೂರ", "ಹಲಸು",
        },
        "vegetables": {
            "ಕ್ಯಾರೆಟ್", "ಆಲೂಗಡ್ಡೆ", "ಟೊಮ್ಯಾಟೊ", "ಈರುಳ್ಳಿ", "ಬೆಳ್ಳುಳ್ಳಿ", "ಎಲೆಕೋಸು", "ಹೂಕೋಸು", "ಪಾಲಕ್",
            "ಸೌತೆಕಾಯಿ", "ಮೆಣಸಿನಕಾಯಿ", "ಬದನೆಕಾಯಿ", "ಕುಂಬಳಕಾಯಿ", "ಮೂಲಂಗಿ", "ಜೋಳ", "ಬಟಾಣಿ", "ಬೀನ್ಸ್",
            "ಬೆಂಡೆಕಾಯಿ", "ಶುಂಠಿ", "ಅಣಬೆ",
        },
        "household items": {
            "ಕುರ್ಚಿ", "ಮೇಜು", "ಮಂಚ", "ದೀಪ", "ಕನ್ನಡಿ", "ಕಂಬಳಿ", "ದಿಂಬು", "ಪೊರಕೆ", "ಬಕೆಟ್", "ಗಡಿಯಾರ",
            "ಪರದೆ", "ಕಾರ್ಪೆಟ್", "ಬೀರು", "ಫ್ಯಾನ್", "ಬುಟ್ಟಿ", "ಏಣಿ", "ಕಸದ ಬುಟ್ಟಿ", "ಹೂದಾನಿ", "ಮೇಣದಬತ್ತಿ",
        },
        "body parts": {
            "ತಲೆ", "ಕಣ್ಣು", "ಕಿವಿ", "ಮೂಗು", "ಬಾಯಿ", "ಕೈ", "ಕಾಲು", "ಬೆರಳು", "ಪಾದ", "ಭುಜ", "ಮೊಣಕೈ",
            "ಮೊಣಕಾಲು", "ಕುತ್ತಿಗೆ", "ಎದೆ", "ಬೆನ್ನು", "ಕೂದಲು", "ಕೆನ್ನೆ", "ನಾಲಿಗೆ", "ಹಲ್ಲು", "ಹೊಟ್ಟೆ",
        },
        "clothing": {
            "ಅಂಗಿ", "ಪ್ಯಾಂಟ್", "ಸೀರೆ", "ಲಂಗ", "ಜಾಕೆಟ್", "ಸ್ವೆಟರ್", "ಟೋಪಿ", "ಕೈಗವಸು", "ಸಾಕ್ಸ್", "ಶೂ",
            "ಚಪ್ಪಲಿ", "ಬೆಲ್ಟ್", "ಟೈ", "ಕೋಟು", "ಶಾಲು", "ಪಂಚೆ", "ಸ್ಕಾರ್ಫ್", "ಬನಿಯನ್",
        },
        "transportation": {
            "ಕಾರು", "ಬಸ್ಸು", "ರೈಲು", "ಸೈಕಲ್", "ಮೋಟಾರ್ ಸೈಕಲ್", "ವಿಮಾನ", "ದೋಣಿ", "ಹಡಗು", "ಲಾರಿ",
            "ಟ್ಯಾಕ್ಸಿ", "ಸ್ಕೂಟರ್", "ಟ್ರಾಕ್ಟರ್", "ಆಟೋ", "ಹೆಲಿಕಾಪ್ಟರ್", "ವ್ಯಾನ್",
        },
        "flowers": {
            "ಗುಲಾಬಿ", "ಲಿಲ್ಲಿ", "ತಾವರೆ", "ಮಲ್ಲಿಗೆ", "ಚೆಂಡು ಹೂ", "ಸೂರ್ಯಕಾಂತಿ", "ಟುಲಿಪ್", "ಆರ್ಕಿಡ್",
            "ದಾಸವಾಳ", "ಡೇಲಿಯಾ",
        },
    },
    "hi": {
        "animals": {
            "कुत्ता", "बिल्ली", "बाघ", "शेर", "घोड़ा", "गाय", "बकरी", "भेड़", "सुअर", "हाथी",
            "जिराफ़", "ज़ेबरा", "बंदर", "भालू", "भेड़िया", "लोमड़ी", "हिरण", "खरगोश", "गिलहरी", "चूहा",
            "चमगादड़", "कंगारू", "पांडा", "ऊंट", "गधा", "भैंस", "गैंडा", "तेंदुआ", "कछुआ", "मेंढक",
            "सांप", "मगरमच्छ", "चील", "उल्लू", "तोता", "गौरैया", "मोर", "बत्तख", "मुर्गी",
        },
        "fruits": {
            "सेब", "केला", "संतरा", "अंगूर", "आम", "अनानास", "स्ट्रॉबेरी", "तरबूज", "पपीता", "अमरूद",
            "कीवी", "आड़ू", "नाशपाती", "चेरी", "नींबू", "नारियल", "अनार", "अंजीर", "कटहल",
        },
        "vegetables": {
            "गाजर", "आलू", "टमाटर", "प्याज़", "लहसुन", "पत्तागोभी", "फूलगोभी", "पालक", "खीरा", "मिर्च",
            "बैंगन", "कद्दू", "मूली", "मक्का", "मटर", "सेम", "भिंडी", "अदरक", "मशरूम",
        },
        "household items": {
            "कुर्सी", "मेज़", "बिस्तर", "दीपक", "आईना", "कंबल", "तकिया", "झाड़ू", "बाल्टी", "घड़ी",
            "पर्दा", "कालीन", "अलमारी", "पंखा", "टोकरी", "सीढ़ी", "कूड़ादान", "फूलदान", "मोमबत्ती",
        },
        "body parts": {
            "सिर", "आँख", "कान", "नाक", "मुँह", "हाथ", "पैर", "उंगली", "तलवा", "कंधा", "कोहनी",
            "घुटना", "गर्दन", "छाती", "पीठ", "बाल", "गाल", "जीभ", "दांत", "पेट",
        },
        "clothing": {
            "कमीज़", "पैंट", "साड़ी", "स्कर्ट", "जैकेट", "स्वेटर", "टोपी", "दस्ताने", "मोज़े", "जूते",
            "चप्पल", "बेल्ट", "टाई", "कोट", "शॉल", "धोती", "कुर्ता", "स्कार्फ़", "बनियान",
        },
        "transportation": {
            "कार", "बस", "ट्रेन", "साइकिल", "मोटरसाइकिल", "हवाई जहाज़", "नाव", "जहाज़", "ट्रक", "टैक्सी",
            "स्कूटर", "ट्रैक्टर", "रिक्शा", "हेलीकॉप्टर", "वैन",
        },
        "flowers": {
            "गुलाब", "लिली", "कमल", "चमेली", "गेंदा", "सूरजमुखी", "ट्यूलिप", "ऑर्किड", "गुड़हल", "डहेलिया",
        },
    },
}

LEVEL3_CATEGORY_KEYS: dict[str, list[str]] = {
    lang: list(labels.keys()) for lang, labels in LEVEL3_CATEGORY_LABELS.items()
}


def level1_word(language: str | None) -> str:
    lang = normalize_language(language)
    return choice(LEVEL1_WORDS.get(lang, LEVEL1_WORDS[DEFAULT_LANGUAGE]))


def level2_sentence(language: str | None) -> str:
    lang = normalize_language(language)
    return choice(LEVEL2_SENTENCES.get(lang, LEVEL2_SENTENCES[DEFAULT_LANGUAGE]))


def level3_category_key(language: str | None) -> str:
    lang = normalize_language(language)
    keys = LEVEL3_CATEGORY_KEYS.get(lang) or LEVEL3_CATEGORY_KEYS[DEFAULT_LANGUAGE]
    return choice(keys)


def level3_category_label(language: str | None, category_key: str) -> str:
    lang = normalize_language(language)
    labels = LEVEL3_CATEGORY_LABELS.get(lang, LEVEL3_CATEGORY_LABELS[DEFAULT_LANGUAGE])
    return labels.get(category_key, category_key)


def category_word_list(language: str | None, category_key: str) -> set[str]:
    lang = normalize_language(language)
    by_category = CATEGORY_WORD_LISTS.get(lang, CATEGORY_WORD_LISTS[DEFAULT_LANGUAGE])
    return by_category.get((category_key or "").strip(), set())


# Extra filler/hesitation words seen in spoken Tamil, Kannada and Hindi,
# merged with the English filler list in voice_analysis.py. Kept small and
# deterministic, same design choice as the original English list.
EXTRA_FILLER_WORDS: dict[str, set[str]] = {
    "ta": {"அதாவது", "அப்படி", "என்னன்னா"},
    "kn": {"ಅಂದರೆ", "ಏನೋ"},
    "hi": {"मतलब", "वो", "अरे", "यानी"},
}

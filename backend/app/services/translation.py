"""Hindi/Hinglish detection, normalization and ONNX translation.

The service answers one question for the API layer: *which text should the
sentiment model actually read?*

    English (default)          -> the original text, unchanged
    Devanagari Hindi           -> the original text (the MT model reads it)
    Romanized Hindi / Hinglish -> normalized to Devanagari, then translated
                                   to English

``Helsinki-NLP/opus-mt-hi-en`` is exported to fp16 ONNX and decoded greedily
on ONNX Runtime; PyTorch is never imported at runtime. Sessions are created
lazily and shared process-wide, exactly like ``app.services.sentiment``, so
English requests never pay for the translation model.

Normalization is lexicon-driven rather than a blind transliteration. The MT
model is trained on clean Devanagari, so ``प्रोडक्ट`` ("procreator") and
``product`` both translate worse than ``उत्पाद``. A curated map therefore
converts the words that matter -- function words, sentiment vocabulary and
frequent English loanwords -- and leaves anything unknown in Latin, which
degrades gracefully instead of producing nonsense.

Public API:
    - ``TranslationResult``    structured outcome for one request
    - ``detect_language()``    "en" | "hi" | "hinglish"
    - ``roman_to_devanagari()`` orthography normalization
    - ``HindiTranslator``      ONNX translation
    - ``get_translator()``     process-wide translator singleton
    - ``prepare_text()``       one-call entry point used by the API
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from app import config

logger = logging.getLogger(__name__)

_INIT_LOCK = threading.Lock()
_TRANSLATOR_LOCK = threading.Lock()

Language = Literal["en", "hi", "hinglish"]
TranslationStatus = Literal["not_needed", "translated", "failed"]

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
_WORD_RE = re.compile(r"[\w\u0900-\u097f']+|[^\w\s]", re.UNICODE)

# Roman-Hindi function words that are never English: one hit is enough to
# classify a Latin-script text as Hinglish.
_STRONG_MARKERS = frozenset(
    {
        "nahi", "nahin", "nhi", "nahín", "hai", "hain", "hoon", "mera", "meri",
        "mere", "mujhe", "muje", "mujhko", "achha", "accha", "acha", "achhi",
        "achi", "achhe", "kharab", "kharabi", "bekar", "bekaar", "bakwas",
        "bakwaas", "pasand", "bilkul", "gussa", "gusse", "kyun", "kaise",
        "kaisey", "kab", "kaun", "kahan", "chahiye", "chahye", "dheere",
        "jaldi", "thoda", "thodi", "jyada", "zyada", "kratrim", "kritrim",
        "budhi", "buddhi", "budhimatta", "buddhimatta", "zabardast", "kamaal",
        "mast", "maza", "shandar", "khoobsurat", "ghatiya", "chokar", "pyaar",
        "pyar", "prem", "mohabbat", "yaar", "bhai", "behen", "dost", "dosto",
        "yaaron", "saukh", "sukoon", "shukriya", "dhanyavad", "khoob", "sasta",
        "sasti", "mehnga", "mahnga", "faltu", "bura", "buri", "naraz",
        "udaas", "udaasi", "pareshan", "pareshani", "chinta", "takleef",
        "dukh", "dard", "badhiya", "mubarak", "shubh", "salaam", "namaste",
        "waqt", "zamana", "zindagi", "duniya", "jivan", "parivaar", "shaadi",
        "jhelna", "chhuti", "tyohaar", "janamdin", "khana", "sona", "mausam",
        "barish", "baarish", "thanda", "bhojan", "chai", "paani", "doodh",
        "kripya", "dhanyavaad", "namaskar",
    }
)

# Weaker markers; two or more of these also mean Hinglish.
#
# Proper nouns and generic English words are deliberately absent: a person's
# name ("Mukesh", "Vijay") and words like "name" occur in romanized Hindi but
# are not evidence of it, so including them mislabeled ordinary English.
_SUPPORT_MARKERS = frozenset(
    {
        "hu", "hun", "thi", "tha", "raha", "rahi", "rahe", "rha",
        "kar", "karta", "karti", "karte", "karna", "kiya", "kiye", "ki", "ka",
        "ke", "ko", "se", "par", "pe", "phir", "fir", "ab", "abhi", "sath",
        "saath", "tak", "liye", "wala", "wali", "wale", "waala", "aap",
        "aapko", "aapki", "aapka", "tum", "tumhe", "tumhari", "tera", "teri",
        "tere", "ham", "hamara", "hamari", "bahut", "bohot", "bahot", "boht",
        "bhi", "hi", "aur", "kya", "kyan", "laga", "lagi", "lagta", "lagti",
        "lage", "gaya", "gayi", "aaya", "aayi", "aaye", "rahta", "rahti",
        "hota", "hoti", "hona", "mein", "main", "mai", "ne", "log", "waqt-en",
        "nam", "naam", "jhel", "padh", "padhai", "likh",
        "samajh", "dhundh", "dhundho", "dekho", "suno", "batao", "chalo",
        "kripya-en", "shayad", "jab", "tab", "yaha", "waha", "yahan", "wahan",
        "idhar", "udhar", "jab-en", "jahan", "kaise-en", "kyun-en",
    }
)

# Ordinary English words that are not romanized-Hindi function words. Used only
# to veto a Hinglish verdict that rests on weak markers alone, so that a plain
# English sentence which merely contains a Hindi-looking word ("name") or a
# person's name ("Mukesh") is not dragged through the translator.
_ENGLISH_MARKERS = frozenset(
    {
        # determiners, pronouns, question words
        "the", "a", "an", "my", "your", "his", "her", "our", "their", "this",
        "that", "these", "those", "i", "me", "you", "he", "she", "it", "we",
        "they", "who", "whom", "whose", "which", "what", "some", "any", "each",
        "every", "much", "many", "most", "few", "all", "both", "other",
        # prepositions, conjunctions, auxiliaries, adverbs
        "of", "in", "on", "at", "to", "for", "from", "with", "without", "by",
        "as", "until", "while", "than", "then", "so", "but", "and", "or",
        "because", "although", "though", "however", "is", "am", "are", "was",
        "were", "be", "been", "being", "do", "does", "did", "have", "has",
        "had", "will", "would", "can", "could", "should", "shall", "may",
        "might", "must", "not", "no", "yes", "there", "here", "now", "never",
        "always", "very", "too", "also", "just", "still", "about", "after",
        "before", "again", "only", "even", "better", "worse",
        # everyday nouns that are not Hindi function words
        "name", "friend", "friends", "restaurant", "product", "products",
        "order", "delivery", "service", "quality", "price", "staff", "help",
        "support", "package", "box", "day", "days", "night", "time", "person",
        "people", "place", "city", "country", "company", "team", "manager",
        "customer", "experience", "problem", "issue", "movie", "film", "book",
        "phone", "battery", "weather", "food", "waiter", "email", "message",
        "call", "account", "number", "amount", "system", "machine", "app",
    }
)

# English function words and loanwords that must never be transliterated,
# even when they look like Hindi. Checked before ``_HINDI_TOKENS``.
_LATIN_PRESERVE = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "so", "not", "no", "yes",
        "ok", "okay", "i", "you", "we", "they", "he", "she", "it", "this",
        "that", "these", "those", "my", "your", "his", "her", "our", "their",
        "is", "am", "are", "was", "were", "be", "been", "being", "do", "does",
        "did", "will", "would", "can", "could", "should", "shall", "may",
        "might", "must", "have", "has", "had", "very", "too", "also", "just",
        "for", "from", "with", "at", "by", "in", "on", "of", "about",
        "as", "than", "then", "there", "here", "when", "where", "what", "who",
        "why", "how", "all", "any", "some", "each", "more", "most", "much",
        "many", "few", "up", "out", "now", "again", "never", "always",
        "thanks", "thank", "please", "hello", "hey", "bye",
    }
)

# Roman-Hindi -> Devanagari. Curated: function words, sentiment vocabulary and
# loanwords, each written the way the MT model saw it in training.
_HINDI_TOKENS: dict[str, str] = {
    # --- pronouns and determiners
    "mujhe": "मुझे", "muje": "मुझे", "mujhko": "मुझको", "meri": "मेरी",
    "mera": "मेरा", "mere": "मेरे", "ham": "हम", "hamara": "हमारा",
    "hamari": "हमारी", "hamko": "हमको", "humko": "हमको",
    "aap": "आप", "aapko": "आपको", "aapki": "आपकी", "aapka": "आपका",
    "aapke": "आपके", "tum": "तुम", "tumhe": "तुम्हें", "tumhari": "तुम्हारी",
    "tume": "तुम", "tu": "तू", "tera": "तेरा", "teri": "तेरी", "tere": "तेरे",
    "wo": "वह", "woh": "वह", "voh": "वह", "ye": "यह", "yah": "यह",
    "yeh": "यह", "yhi": "यही", "wahi": "वही", "yahi": "यही",
    "uska": "उसका", "uski": "उसकी", "iske": "इसके", "iska": "इसका",
    "iski": "इसकी", "unka": "उनका", "unki": "उनकी", "inka": "इनका",
    "inki": "इनकी", "jiska": "जिसका", "jiski": "जिसकी", "kis": "किस",
    "kisi": "किसी", "kisko": "किसको", "unhe": "उन्हें", "inhe": "इन्हें",
    "sab": "सब", "sabko": "सबको", "sabka": "सबका", "sabki": "सबकी",
    "sabke": "सबके", "sabse": "सबसे", "sabhi": "सभी", "har": "हर",
    "kuch": "कुछ", "kuchh": "कुछ", "koi": "कोई", "dono": "दोनों",
    "sirf": "सिर्फ़", "bas": "बस", "apna": "अपना", "apni": "अपनी",
    "apne": "अपने", "yaha": "यहाँ", "waha": "वहाँ", "yahan": "यहाँ",
    "wahan": "वहाँ", "idhar": "इधर", "udhar": "उधर", "jahan": "जहाँ",
    "yahan-en": "यहाँ", "wahan-en": "वहाँ",
    # --- verbs, copula and auxiliaries
    "hai": "है", "hain": "हैं", "hu": "हूँ", "hoon": "हूँ", "hun": "हूँ",
    "tha": "था", "thi": "थी", "raha": "रहा", "rahi": "रही", "rahe": "रहे",
    "rha": "रह", "rahta": "रहता", "rahti": "रहती", "rahte": "रहते",
    "kar": "कर", "karta": "करता", "karti": "करती", "karte": "करते",
    "karna": "करना", "karke": "करके", "karenge": "करेंगे", "karega": "करेगा",
    "karegi": "करेगी", "kiya": "किया", "kiye": "किये", "kijiyega": "कीजिए",
    "kijiye": "कीजिये", "kijye": "कीजिये", "kiyaa": "किया",
    "hota": "होता", "hoti": "होती", "hona": "होना", "hoga": "होगा",
    "hogi": "होगी", "hote": "होते", "hone": "होने", "huen": "हुएं",
    "gaya": "गया", "gayi": "गई", "gaaya": "गया", "gaayi": "गई",
    "aaya": "आया", "aayi": "आई", "aaye": "आए", "aaen": "आएं",
    "chala": "चला", "chali": "चली", "chalo": "चलो", "aao": "आओ",
    "jao": "जाओ", "jaa": "जा", "jana": "जाना", "aana": "आना",
    "lena": "लेना", "dena": "देना", "leta": "लेता", "leti": "लेती",
    "deta": "देता", "dete": "देते", "lo": "लो", "padho": "पढ़ो",
    "padh": "पढ़", "padhna": "पढ़ना", "likho": "लिखो", "likhna": "लिखना",
    "dekho": "देखो", "dekha": "देखा", "dekhi": "देखी", "dekhna": "देखना",
    "suno": "सुनो", "suna": "सुना", "sunna": "सुनना", "suniye": "सुनिए",
    "batao": "बताओ", "bata": "बता", "bataen": "बताएं", "bataye": "बताए",
    "dhundho": "धूँधो", "dhundh": "धूँध", "dhundha": "धूँधा", "dhundhna": "धूँधना",
    "samajh": "समझ", "samajhna": "समझना", "samjho": "समझो", "samjha": "समझा",
    "bhejo": "भेजो", "bhej": "भेज", "bhejna": "भेजना", "bheja": "भेजा",
    "karo": "करो", "karna-en": "करना", "karna2": "करना",
    # --- postpositions and connectives
    "ki": "की", "ka": "का", "ke": "के", "ko": "को", "se": "से",
    "par": "पर", "pe": "पे", "phir": "फिर", "fir": "फिर", "ab": "अब",
    "abhi": "अब", "sath": "साथ", "saath": "साथ", "tak": "तक",
    "liye": "लिए", "bina": "बिना", "wala": "वाला", "wali": "वाली",
    "wale": "वाले", "waala": "वाला", "waali": "वाली", "waale": "वाले",
    "jaisa": "जैसा", "jaisi": "जैसी", "jaise": "जैसे", "tarah": "तरह",
    "lekin": "लेकिन", "magar": "मगर", "isliye": "इसलिए", "kyunki": "क्योंकि",
    "jab": "जब", "tab": "तब", "to": "तो", "hi": "ही", "bhi": "भी",
    "aur": "और", "ya": "या", "nahi": "नहीं", "nahin": "नहीं", "nhi": "नहीं",
    "na": "ना", "kabhi": "कभी", "mein": "मैं", "ne": "ने", "log": "लोग",
    # --- degree, quantity and time
    "bahut": "बहुत", "bohot": "बहुत", "bahot": "बहुत", "boht": "बहुत",
    "vahut": "बहुत", "thoda": "थोड़ा", "thodi": "थोड़ी", "jyada": "ज़्यादा",
    "zyada": "ज़्यादा", "kam": "कम", "bilkul": "बिल्कुल", "poora": "पूरा",
    "pura": "पूरा", "puri": "पूरी", "poori": "पूरी", "adhoora": "अधूरा",
    "aaj": "आज", "kal": "कल", "parso": "परसों", "raat": "रात",
    "subah": "सुबह", "sham": "शाम", "dopahar": "दोपहर", "samay": "समय",
    "waqt": "वक्त", "time": "समय", "din": "दिन", "hafte": "हफ़्ते",
    "mahine": "महीने", "saal": "साल", "umar": "उम्र", "jaldi": "जल्दी",
    "dheere": "धीरे", "aaram": "आराम", "jaldi-en": "जल्दी", "bada": "बड़ा",
    "chota": "छोटा", "purana": "पुराना", "naya": "नया", "nayi": "नयी",
    # --- question words
    "kya": "क्या", "kyun": "क्यों", "kyan": "क्यों", "kab": "कब",
    "kaun": "कौन", "kahan": "कहाँ", "kaha": "कहाँ", "kaise": "कैसे",
    "kaisa": "कैसा", "kaisi": "कैसी", "kaisey": "कैसे", "kitna": "कितना",
    "kitni": "कितनी", "kitne": "कितने", "shayad": "शायद",
    # --- sentiment: positive
    "achha": "अच्छा", "accha": "अच्छा", "acha": "अच्छा", "achhi": "अच्छी",
    "achi": "अच्छी", "ache": "अच्छे", "achhe": "अच्छे", "acche": "अच्छे",
    "khoob": "खूब", "khoobsurat": "खूबसूरत", "shandar": "शानदार",
    "badhiya": "बढ़िया", "mast": "मास्ट", "kamaal": "कमाल", "maza": "मज़ा",
    "sundar": "सुंदर", "lalit": "ललित", "rangeen": "रंगीन", "sasta": "सस्ता",
    "sasti": "सस्ती", "pasand": "पसंद", "anpasand": "अनपसंद",
    "khush": "खुश", "khushi": "खुशी", "sukh": "सुख", "sukoon": "सुकून",
    "saukh": "सौख", "shant": "शांत", "chain": "चैन", "pyaar": "प्यार",
    "pyar": "प्यार", "prem": "प्रेम", "mohabbat": "मोहब्बत", "waada": "वादा",
    "shukriya": "शुक्रिया", "dhanyavad": "धन्यवाद", "shukr": "शुक्र",
    "prashansha": "प्रशंसा", "badshah": "बढ़शाह", "shabash": "शाबाश",
    "kripya": "कृपया", "maaf": "माफ़", "maafi": "माफ़ी", "shubh": "शुभ",
    "namaste": "नमस्ते", "salaam": "सलाम", "mubarak": "मुबारक",
    "vijay": "विजय", "shubham": "शुभम", "dhanyavaad": "धन्यवाद",
    "namaskar": "नमस्कार", "vishwas": "विश्वास", "bharosa": "भरोसा",
    "imaandari": "ईमानदारी", "sachchai": "सच्चाई", "sach": "सच",
    "jhooth": "झूठ", "jhut": "झूठ", "sabse-accha": "सबसे अच्छा",
    # --- sentiment: negative
    "kharab": "खराब", "kharabi": "खराबी", "bekar": "बेकार", "bekaar": "बेकार",
    "bakwas": "बकवास", "bakwaas": "बकवास", "faltu": "फ़ालतू",
    "bura": "बुरा", "bur": "बुरा", "buri": "बुरी", "bure": "बुरे",
    "ghatiya": "घटिया", "chokar": "छोकर", "gussa": "गुस्सा", "gusse": "गुस्से",
    "naraz": "नाराज़", "nirgusa": "निर्गुसा", "dukh": "दुख", "dard": "दर्द",
    "udaas": "उदास", "udaasi": "उदासी", "pareshan": "परेशान",
    "pareshani": "परेशानी", "chinta": "चिंता", "takleef": "तकलीफ़",
    "tension": "टेंशन", "stress": "तनाव", "pressure": "दबाव",
    "complaint": "शिकायत", "shikayat": "शिकायत", "kami": "कमी",
    "galat": "गलत", "durd": "दुर्द", "musibat": "मुसीबत",
    "neh": "नेह", "majboor": "मजबूर", "majbur": "मजबूर",
    # --- common nouns: people, home, time
    "ghar": "घर", "parivaar": "परिवार", "dost": "दोस्त", "dosto": "दोस्तों",
    "yaar": "यार", "yaaron": "यारों", "bhai": "भाई", "behen": "बहन",
    "didi": "दीदी", "bhaiya": "भैया", "maa": "माँ", "papa": "पापा",
    "beta": "बेटा", "beti": "बेटी", "pati": "पति", "patni": "पत्नी",
    "naam": "नाम", "nam": "नाम", "name": "नाम", "mukesh": "मुकेश",
    "log": "लोग", "logo": "लोगों", "insaan": "इंसान", "customer": "ग्राहक",
    "teacher": "शिक्षक", "student": "छात्र", "baap": "बाप", "maa-en": "माँ",
    "shaadi": "शादी", "vivah": "विवाह", "function": "फ़ंक्शन",
    "party": "पार्टी", "janam": "जन्म", "janamdin": "जन्मदिन",
    "birthday": "बर्थडे", "tyohaar": "त्योहार", "chhuti": "छुट्टी",
    "holiday": "हॉलिडे", "kaam": "काम", "office": "ऑफ़िस", "school": "स्कूल",
    "college": "कॉलेज", "hospital": "अस्पताल", "doctor": "डॉक्टर",
    "police": "पुलिस", "sarkar": "सरकार", "yojana": "योजना", "pata": "पता",
    "baat": "बात", "kahani": "कहानी", "sawaal": "सवाल", "jawaab": "जवाब",
    "jivan": "जीवन", "zindagi": "ज़िंदगी", "duniya": "दुनिया",
    "zamana": "ज़माना", "jagah": "जगह", "man": "मन", "dil": "दिल",
    "himmat": "हिम्मत", "taqat": "ताक़त", "jab-en": "जब",
    # --- very frequent Hinglish verbs and nouns still in daily use
    "main": "मैं", "mai": "मैं", "mujhhe": "मुझे", "hamko-en": "हमको",
    "karne": "करने", "karenge-en": "करेंगे", "karna-2": "करना",
    "mila": "मिला", "mili": "मिली", "milta": "मिलता", "milti": "मिलती",
    "mile": "मिले", "milenge": "मिलेंगे", "laga-2": "लगा",
    "bimar": "बीमार", "tabiyat": "तबीयत", "tabyat": "तबीयत",
    "ghanta": "घंटा", "ghante": "घंटे", "ghanti": "घंटी", "saat": "घंटी",
    "doodh-en": "दूध", "roti-en": "रोटी", "makai": "मकई", "gehun": "गेहूँ",
    "kisan": "किसान", " fasal": "फसल", "kisan-en": "किसान",
    "warranty": "वारंटी", "guarantee": "गारंटी", "claim": "दावा",
    "shop": "दुकान", "dukan": "दुकान", "market": "बाज़ार", "bazaar": "बाज़ार",
    "bill": "बिल", "receipt": "रसीद", "invoice": "इनवॉइस", "coupon": "कूपन",
    "helpline": "हेल्पलाइन", "complain": "शिकायत",
    "complained": "शिकायत", "report": "रिपोर्ट", "rating": "रेटिंग",
    "review": "समीक्षा", "star": "स्टार", "stars": "स्टार", "point": "प्वाइंट",
    "life": "लाइफ़", "backup": "बैकअप", "performance": "परफ़ॉर्मेंस",
    "speed": "स्पीड", "faster": "तेज़", "slower": "धीमा", "lighter": "हल्का",
    "helpful": "मददगार", "perfect": "परफ़ेक्ट", "useless": "बेकार",
    "worth": "वर्थ", "value": "वैल्यू", "genuine": "असली",
    "original": "ऑरिजिनल", "duplicate": "डुप्लीकेट", "waste": "वेस्ट",
    "wasted": "वेस्ट", "expensive": "महँगा", "cheap": "सस्ता",
    "affordable": "किफ़ायती", "premium": "प्रीमियम", "basic": "बेसिक",
    "standard": "स्टैंडर्ड", "girl": "लड़की", "boy": "लड़का",
    "ladka": "लड़का", "ladki": "लड़की", "aadmi": "आदमी", "aurat": "औरत",
    "janta": "जनता", "bheed": "भीड़", "kitab": "किताब", "kagaz": "कागज़",
    "pen": "पेन", "pencil": "पेंसिल", "bag": "बैग", "kursi": "कुर्सी",
    "darwaza": "दरवाज़ा", "khidki": "खिड़की", "deewar": "दीवार",
    "paani-en": "पानी", "dhoop": "धूप", "chand": "चाँद", "aasmaan": "आसमान",
    # --- common nouns: food, weather, daily life
    "mausam": "मौसम", "barish": "बारिश", "baarish": "बारिश", "thanda": "ठंडा",
    "thand": "ठंड", "garmi": "गर्मी", "bhojan": "भोजन", "khana": "खाना",
    "khaana": "खाना", "roti": "रोटी", "chapati": "चपाती", "dal": "दाल",
    "sabzi": "सब्ज़ी", "chai": "चाय", "paani": "पानी", "pani": "पानी",
    "doodh": "दूध", "honey": "शहद", "namkeen": "नमकीन", "biryani": "बिरयानी",
    "samosa": "समोसा", "dosa": "डोसा", "idli": "इडली", "halwa": "हलवा",
    "laddoo": "लड्डू", "mithai": "मिठाई", "jhelna": "झेलना", "khelna": "खेलना",
    "khelo": "खेलो", "padhaaya": "पढ़ाया", "padhai": "पढ़ाई", "exam": "परीक्षा",
    "soja": "सो जा", "sona": "सोना", "peena": "पीना", "udna": "उड़ना",
    # --- English loanwords, written the way the MT model knows them
    "product": "उत्पाद", "products": "उत्पाद", "quality": "गुणवत्ता",
    "service": "सेवा", "services": "सेवा", "price": "कीमत", "order": "ऑर्डर",
    "delivery": "डिलीवरी", "support": "सहायता", "company": "कंपनी",
    "team": "टीम", "staff": "स्टाफ़", "phone": "फ़ोन", "phones": "फ़ोन",
    "mobile": "मोबाइल", "laptop": "लैपटॉप", "screen": "स्क्रीन",
    "battery": "बैटरी", "camera": "कैमरा", "app": "ऐप", "software": "सॉफ़्टवेयर",
    "network": "नेटवर्क", "internet": "इंटरनेट", "website": "वेबसाइट",
    "server": "सर्वर", "system": "सिस्टम", "user": "उपयोगकर्ता",
    "users": "उपयोगकर्ता", "brand": "ब्रांड", "size": "आकार", "color": "रंग",
    "movie": "फ़िल्म", "movies": "फ़िल्म", "film": "फ़िल्म", "films": "फ़िल्म",
    "music": "संगीत", "song": "गाना", "video": "वीडियो", "game": "खेल",
    "food": "खाना", "coffee": "कॉफ़ी", "tea": "चाय", "pizza": "पिज़्ज़ा",
    "burger": "बर्गर", "car": "गाड़ी", "train": "ट्रेन", "bus": "बस",
    "bike": "बाइक", "tv": "टीवी", "watch": "घड़ी", "box": "बक्स",
    "headphones": "हेडफ़ोन", "speaker": "स्पीकर", "charger": "चार्जर",
    "cable": "केबल", "sound": "ध्वनि", "experience": "अनुभव", "crack": "टूट",
    "damage": "नुकसान", "defect": "खराबी", "problem": "समस्या", "issue": "समस्या",
    "behaviour": "व्यवहार", "response": "जवाब", "payment": "भुगतान",
    "charge": "शुल्क", "money": "पैसा", "paisa": "पैसा", "paise": "पैसे",
    "rupees": "रुपये", "offer": "ऑफ़र", "discount": "छूट", "sale": "बिक्री",
    "refund": "धनवापसी", "return": "वापस", "exchange": "बदली",
    "recommend": "सिफ़ारिश", "reply": "उत्तर", "wait": "इंतज़ार",
    "waiting": "इंतज़ार", "delay": "देरी", "late": "देर", "fast": "तेज़",
    "slow": "धीमा", "good": "अच्छा", "bad": "बुरा", "best": "सबसे अच्छा",
    "worst": "सबसे बुरा", "great": "बहुत अच्छा", "nice": "अच्छा",
    "love": "प्यार", "hate": "नफ़रत", "like": "पसंद", "want": "चाहिए",
    "need": "चाहिए", "help": "मदद", "thanks-en": "धन्यवाद",
    "work": "काम", "works": "काम", "working": "काम", "money-en": "पैसा",
    "booking": "बुकिंग", "ticket": "टिकट", "hotel": "होटल", "food-en": "खाना",
    "seat": "सीट", "room": "कमरा", "water": "पानी", "wash": "धुलाई",
    "clean": "साफ़", "dirty": "गंदा", "fresh": "ताज़ा", "cold": "ठंडा",
    "hot": "गर्म", "ready": "तैयार", "open": "खुला", "close": "बंद",
    "start": "शुरू", "finish": "ख़त्म", "sorry": "माफ़", "welcome": "स्वागत",
}


@dataclass(frozen=True)
class TranslationResult:
    """Outcome of preparing one request for the sentiment model.

    ``text_for_model`` is what the sentiment analyzer should read;
    ``translated_text`` is the English rendering, or ``None`` when no
    translation was needed (or when translation failed).
    """

    original_text: str
    text_for_model: str
    translated_text: str | None
    language: Language
    status: TranslationStatus


def contains_devanagari(text: str) -> bool:
    """True if ``text`` contains any Devanagari character."""
    return bool(_DEVANAGARI_RE.search(text))


def _tokenize(text: str) -> list[str]:
    """Split into word and punctuation tokens, preserving both for reassembly."""
    return _WORD_RE.findall(text)


def detect_language(text: str) -> Language:
    """Classify ``text`` as "en", "hi" (Devanagari) or "hinglish".

    Deliberately conservative: English text must never be mislabeled, so a
    Latin-script text needs either one unambiguous marker or two weak ones.
    """
    if not text or not text.strip():
        return "en"
    if contains_devanagari(text):
        return "hi"

    words = [w.lower() for w in _tokenize(text)]
    strong = sum(1 for w in words if w in _STRONG_MARKERS)
    if strong >= 1:
        return "hinglish"

    support = sum(1 for w in words if w in _SUPPORT_MARKERS)
    if support >= 2:
        # Weak markers alone are not conclusive, because several of them are
        # also ordinary English ("name", "no") or appear in sentences that are
        # plainly English. Require the sentence to not read as English before
        # treating weak markers as Hinglish.
        english = sum(1 for w in words if w in _ENGLISH_MARKERS)
        if english < 2 or english < support:
            return "hinglish"
    return "en"


def _normalize_token(token: str) -> str:
    """Return the Devanagari form of ``token``, or the token itself."""
    if not token.isascii() or not token.isalpha():
        return token  # punctuation, digits, emoji, already-Devanagari
    if token.lower() in _LATIN_PRESERVE:
        return token
    return _HINDI_TOKENS.get(token.lower(), token)


def _reflow(pieces: list[str]) -> str:
    """Join tokens with spaces, keeping punctuation attached to its neighbour."""
    text = " ".join(pieces)
    text = re.sub(r"\s+([,.;:!?%\u0964)\]}])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    return text.strip()


def roman_to_devanagari(text: str) -> str:
    """Normalize Romanized Hindi/Hinglish ``text`` into Devanagari.

    Word-by-word so mixed text keeps its English tokens, spacing and
    punctuation. Tokens absent from the map stay in Latin rather than being
    transliterated into a spelling the MT model never saw.
    """
    return _reflow([_normalize_token(t) for t in _tokenize(text)])


class HindiTranslator:
    """Greedy Hindi -> English translation on fp16 ONNX graphs.

    The encoder runs once per input; the decoder is re-evaluated over the
    growing prefix (no KV cache), which keeps the shipped asset list to two
    graphs at a small latency cost on short texts.
    """

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self._model_dir = (
            Path(model_dir) if model_dir is not None else Path(config.TRANSLATION_DIR)
        )
        self._encoder: Any = None
        self._decoder: Any = None
        self._tokenizer: Any = None
        self._start_id: int = 0
        self._eos_id: int = 0

    @property
    def loaded(self) -> bool:
        """True once both ONNX sessions and the tokenizer exist."""
        return self._encoder is not None and self._decoder is not None

    def load(self) -> "HindiTranslator":
        """Create the ONNX sessions and tokenizer (idempotent, thread-safe)."""
        if self.loaded:
            return self
        with _INIT_LOCK:
            if self.loaded:
                return self
            import onnxruntime
            from transformers import AutoTokenizer

            encoder_path = self._model_dir / "encoder_model.onnx"
            decoder_path = self._model_dir / "decoder_model.onnx"
            missing = [str(p) for p in (encoder_path, decoder_path) if not p.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Translation model not found: {', '.join(missing)}. "
                    "Run scripts/prepare_translation.py to export it first."
                )
            options = onnxruntime.SessionOptions()
            options.graph_optimization_level = (
                onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._encoder = onnxruntime.InferenceSession(
                str(encoder_path), options, providers=["CPUExecutionProvider"]
            )
            self._decoder = onnxruntime.InferenceSession(
                str(decoder_path), options, providers=["CPUExecutionProvider"]
            )
            self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
            self._start_id = int(self._tokenizer.pad_token_id or 0)
            self._eos_id = int(self._tokenizer.eos_token_id or 0)
            logger.info(
                "HindiTranslator loaded model from %s (%s)",
                self._model_dir,
                config.TRANSLATION_MODEL_NAME,
            )
        return self

    def translate(self, text: str) -> str:
        """Translate one Devanagari (or normalized) sentence to English."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}.")
        text = text.strip()
        if not text:
            raise ValueError("text must be a non-empty string.")
        self.load()

        encoded = self._tokenizer(
            text,
            max_length=config.TRANSLATION_MAX_SOURCE_TOKENS,
            truncation=True,
        )["input_ids"]
        input_ids = np.asarray([encoded], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)

        encoder_outputs = self._encoder.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        hidden = np.asarray(encoder_outputs[0], dtype=np.float32)

        generated = [self._start_id]
        for _ in range(config.TRANSLATION_MAX_NEW_TOKENS):
            decoder_outputs = self._decoder.run(
                None,
                {
                    "input_ids": np.asarray([generated], dtype=np.int64),
                    "encoder_hidden_states": hidden,
                    "encoder_attention_mask": attention_mask,
                },
            )
            logits = np.asarray(decoder_outputs[0][0, -1], dtype=np.float32)
            next_id = int(np.argmax(logits))
            if next_id == self._eos_id:
                break
            generated.append(next_id)

        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


_translator: HindiTranslator | None = None


def get_translator() -> HindiTranslator:
    """Return the process-wide translator; created once, loaded on first use."""
    global _translator
    if _translator is None:
        with _TRANSLATOR_LOCK:
            if _translator is None:
                _translator = HindiTranslator()
    return _translator


def prepare_text(text: str) -> TranslationResult:
    """Return the text the sentiment model should read, plus its provenance.

    English input is returned untouched with status ``not_needed`` and never
    touches the translation model. Devanagari input is translated as-is;
    Romanized Hindi is normalized first. Any translation error is logged and
    degraded to analyzing the original text, so a missing or broken
    translation model never fails an otherwise valid prediction.
    """
    if not config.TRANSLATION_ENABLED:
        return TranslationResult(
            original_text=text,
            text_for_model=text,
            translated_text=None,
            language="en",
            status="not_needed",
        )

    language = detect_language(text)
    if language == "en":
        return TranslationResult(
            original_text=text,
            text_for_model=text,
            translated_text=None,
            language="en",
            status="not_needed",
        )

    source = roman_to_devanagari(text) if language == "hinglish" else text
    try:
        translated = get_translator().translate(source).strip()
    except Exception as exc:  # missing assets, ONNX error, tokenizer failure
        logger.warning("Hindi translation failed (%s); using original text", exc)
        return TranslationResult(
            original_text=text,
            text_for_model=text,
            translated_text=None,
            language=language,
            status="failed",
        )
    if not translated:
        logger.warning("Hindi translation produced no text; using original text")
        return TranslationResult(
            original_text=text,
            text_for_model=text,
            translated_text=None,
            language=language,
            status="failed",
        )
    return TranslationResult(
        original_text=text,
        text_for_model=translated,
        translated_text=translated,
        language=language,
        status="translated",
    )

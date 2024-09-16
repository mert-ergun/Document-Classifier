import os
import json

def load_translations(lang_code):
    file_path = os.path.join('locales', f'{lang_code}.json')
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def get_translations():
    translations = {
        'en': load_translations('en'),
        'tr': load_translations('tr')
    }
    return translations

translations = get_translations()

def t(key, lang_code):
    return translations[lang_code].get(key, key)
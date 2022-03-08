"""The Endpoints to manage the BOOK_REQUESTS"""
import uuid
from datetime import datetime, timedelta
from flask import jsonify, abort, request, Blueprint, render_template
#from flask import Flask, flash, Response, redirect, url_for, request, session, abort, render_template, make_response, jsonify

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
from validate_email import validate_email
REQUEST_API = Blueprint('request_api', __name__)
    
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(device)

languages = ["Afrikaans", "Amharic", "Arabic", "Asturian", "Azerbaijani", "Bashkir", "Belarusian", "Bulgarian", "Bengali", "Breton", "Bosnian", "Valencian", "Cebuano", "Czech", "Welsh", "Danish", "German", "Greeek", "English", "Spanish", "Estonian", "Persian", "Fulah", "Finnish", "French", "Irish", "Scottish Gaelic", "Galician", "Gujarati", "Hausa", "Hebrew", "Hindi", "Croatian", "Haitian Creole", "Hungarian", "Armenian", "Indonesian", "Igbo", "Iloko", "Icelandic", "Italian", "Japanese", "Javanese", "Georgian", "Kazakh", "Central Khmer", "Kannada", "Korean", "Letzeburgesch", "Ganda", "Lingala", "Lao", "Lithuanian", "Latvian", "Malagasy", "Macedonian", "Malayalam", "Mongolian", "Marathi", "Malay", "Burmese", "Nepali", "Flemish", "Norwegian", "Northern Sotho", "Occitan", "Oriya", "Punjabi", "Polish", "Pashto", "Portuguese", "Moldovan", "Russian", "Sindhi", "Sinhalese", "Slovak", "Slovenian", "Somali", "Albanian", "Serbian", "Swati", "Sundanese", "Swedish", "Swahili", "Tamil", "Thai", "Tagalog", "Tswana", "Turkish", "Ukrainian", "Urdu", "Uzbek", "Vietnamese", "Wolof", "Xhosa", "Yiddish", "Yoruba", "Chinese", "Zulu"]
langslow = (map(lambda x: x.lower(), languages))
langCodes = ["af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", "bs", "ca", "ceb", "cs", "cy", "da", "de", "el", "en", "es", "et", "fa", "ff", "fi", "fr", "ga", "gd", "gl", "gu", "ha", "he", "hi", "hr", "ht", "hu", "hy", "id", "ig", "ilo", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "lb", "lg", "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "my", "ne", "nl", "no", "ns", "oc", "or", "pa", "pl", "ps", "pt", "ro", "ru", "sd", "si", "sk", "sl", "so", "sq", "sr", "ss", "su", "sv", "sw", "ta", "th", "tl", "tn", "tr", "uk", "ur", "uz", "vi", "wo", "xh", "yi", "yo", "zh", "zu"]
langDict = dict(zip(langslow,langCodes))

def get_blueprint():
    """Return the blueprint for the main app module"""
    return REQUEST_API


@REQUEST_API.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")
    

@REQUEST_API.route('/translate', methods=['POST'])
def translate():
    # print( request.form['data'])
    # print(request.method, request.form['sourcelang'])

    if not request.form:
        abort(400)
    
    source_text = request.form['rawtext']
    # print(langDict)
    source_l = langDict[request.form['sourcelang'].lower()]
    if(source_l == ''):
        tokenizer.src_lang = "en"
    else:
        tokenizer.src_lang = source_l
    
    target_l = langDict[request.form['targetlang'].lower()]
    # print(target_l)
    if(target_l == ''):
        target_l = "de"
    else:
        tokenizer.tgt_lang = target_l
    
    encoded_hi = tokenizer([source_text], return_tensors="pt", padding=True).to(device)
    generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id(target_l))
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    # print(translation)
    result = {
        "translated_text": translation[0]
    }
    return jsonify(result), 200

 

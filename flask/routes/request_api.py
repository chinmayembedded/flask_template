"""The Endpoints to manage the BOOK_REQUESTS"""
import uuid
from datetime import datetime, timedelta
from flask import jsonify, abort, request, Blueprint
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
from validate_email import validate_email
REQUEST_API = Blueprint('request_api', __name__)


def get_blueprint():
    """Return the blueprint for the main app module"""
    return REQUEST_API


BOOK_REQUESTS = {
    "8c36e86c-13b9-4102-a44f-646015dfd981": {
        'title': u'Good Book',
        'email': u'testuser1@test.com',
        'timestamp': (datetime.today() - timedelta(1)).timestamp()
    },
    "04cfc704-acb2-40af-a8d3-4611fab54ada": {
        'title': u'Bad Book',
        'email': u'testuser2@test.com',
        'timestamp': (datetime.today() - timedelta(2)).timestamp()
    }
}

'''
@REQUEST_API.route('/request', methods=['GET'])
def get_records():
    """Return all book requests
    @return: 200: an array of all known BOOK_REQUESTS as a \
    flask/response object with application/json mimetype.
    """
    return jsonify(BOOK_REQUESTS)


@REQUEST_API.route('/request/<string:_id>', methods=['GET'])
def get_record_by_id(_id):
    """Get book request details by it's id
    @param _id: the id
    @return: 200: a BOOK_REQUESTS as a flask/response object \
    with application/json mimetype.
    @raise 404: if book request not found
    """
    if _id not in BOOK_REQUESTS:
        abort(404)
    return jsonify(BOOK_REQUESTS[_id])
'''

@REQUEST_API.route('/translate', methods=['POST'])
def translate():
    data = request.get_json(force=True)
    source_l = "en"
    target_l = "de"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B").to(device)
    
    if not data.get['rawtext']:
        abort(400)
    
    source_text = data['rawtext']

    source_l = data['sourcelang'].lower()
    if(source_l == ''):
        tokenizer.src_lang = "en"
    else:
        tokenizer.src_lang = source_l
    
    target_l = data['targetlang'].lower()
    if(target_l == ''):
        target_l = "de"
    
    encoded_hi = tokenizer([source_text], return_tensors="pt", padding=True).to(device)
    generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id(target_l))
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    result = {
        'translated_text':translation[0]
    }
    
    '''
    if not request.get_json():
        abort(400)
    data = request.get_json(force=True)

    if not data.get('email'):
        abort(400)
    if not validate_email(data['email']):
        abort(400)
    if not data.get('title'):
        abort(400)

    new_uuid = str(uuid.uuid4())
    book_request = {
        'title': data['title'],
        'email': data['email'],
        'timestamp': datetime.now().timestamp()
    }
    BOOK_REQUESTS[new_uuid] = book_request
    # HTTP 201 Created
    return jsonify({"id": new_uuid}), 201
    '''



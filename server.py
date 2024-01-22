import sys
from transformers import CLIPTokenizer
from flask import Flask, request, jsonify
import espm

tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')

app = Flask(__name__)

@app.route('/token-api/count/', methods=['POST'])
def add_message():
    content = request.json
    prompt = content['prompt']    
    return jsonify(tokenizer.tokenize(prompt))

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=espm.port,debug=False)
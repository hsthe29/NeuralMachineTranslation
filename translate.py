import json
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
from src.model.language import Language
from src.model.translator import Translator
from src.utils import *
import sys

HOST_NAME = 'localhost'
PORT = 8000


def load_model(ckpt):
    en_vocab = load_vocab('vocab/vocab.en')
    vi_vocab = load_vocab('vocab/vocab.vi')

    special_tokens = get_special_tokens()

    en_processor = Language(en_vocab, special_tokens, lang='en')
    vi_processor = Language(vi_vocab, special_tokens, lang='vi')

    pretrained_model = Translator(en_processor, vi_processor, config)
    pretrained_model.load_weights(ckpt)
    print('create')
    return pretrained_model


class TranslateServer(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        print('path', self.path)
        if self.path == '/':
            self.path = '/index.html'
        try:
            split_path = os.path.splitext(self.path)
            request_extension = split_path[1]
            print(split_path)
            if request_extension != ".py":
                f = open('src/ui' + self.path).read()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(bytes(f, 'utf-8'))
                print('done')
            else:
                f = "File not found" + ''.join(split_path)
                self.send_error(404, f)

        except Exception:
            f = "File not found"
            self.send_error(404, f)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself
        if post_data == b'terminate':
            print('Session terminating...')
            sys.exit(0)
        post_data = json.loads(post_data.decode())
        target_text = translate(model, post_data['text'])
        response_data = {
            'lang': 'vi',
            'text': target_text.decode()
        }
        response_data = json.dumps(response_data).encode('utf=8')
        self._set_response()
        response = BytesIO()
        response.write(response_data)
        self.wfile.write(response.getvalue())


def translate(pretrained_model, text):
    result = pretrained_model.translate(text, max_len=50)
    result_texts = result['text']
    return result_texts[0].numpy()


if __name__ == "__main__":
    model = load_model('checkpoint/translator_v1.tf')

    httpd = HTTPServer((HOST_NAME, PORT), TranslateServer)
    print(time.asctime(), "Start Server - %s:%s" % (HOST_NAME, PORT))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print(time.asctime(), 'Stop Server - %s:%s' % (HOST_NAME, PORT))

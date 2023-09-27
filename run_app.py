import json
import os
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO

from thehs import Translator, Parser
from thehs.model import AutoModel, AutoConfig

parser = Parser()

parser.DEFINE_integer("PORT", default=8000, hint="App's port")
parser.DEFINE_string("config-file", "config.json")

HOST_NAME = 'localhost'


class AppServer(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == '/':
            self.path = '/index.html'
        try:
            split_path = os.path.splitext(self.path)
            request_extension = split_path[1]
            if request_extension != ".py":
                with open('thehs/webui' + self.path, encoding='utf-8') as fi:
                    f = fi.read()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(bytes(f, 'utf-8'))
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
        target_text = translator.translate(post_data['text'])
        response_data = {
            'lang': 'vi',
            'text': target_text[0]
        }
        response_data = json.dumps(response_data).encode('utf=8')
        self._set_response()
        response = BytesIO()
        response.write(response_data)
        self.wfile.write(response.getvalue())


def create_translator():
    config = AutoConfig.from_file(flags.config_file)
    model = AutoModel.from_pretrained(config.name())
    return Translator(config, model)


if __name__ == "__main__":
    flags = parser.parse()

    translator = create_translator()

    httpd = HTTPServer((HOST_NAME, flags.PORT), AppServer)
    print(time.asctime(), "| Start Server - %s:%s" % (HOST_NAME, flags.PORT))
    appURL = f'http://{HOST_NAME}:{flags.PORT}'
    try:
        print(f"Auto redirect to link {appURL}. If not, go click.")
        webbrowser.open(appURL, new=1)
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\033[91mErrors occur when running application. Interrupt!\033[0m")
    finally:
        httpd.server_close()
    print(time.asctime(), "| Stop Server - %s:%s" % (HOST_NAME, flags.PORT))

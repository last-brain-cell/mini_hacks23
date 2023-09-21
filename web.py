import base64

from flask import Flask, request, jsonify

import generate
from generateold import generate_captions

app = Flask(__name__)


@app.route("/test", methods=["GET"])
def main2():
    return "gand mara madhar chod"

@app.route("/", methods=["POST"])
def main():
    print(request.data)
    # with open('sample.jpg', "wb") as fh:
    #     fh.write(base64.decodebytes(request.data))
    #     return jsonify({
    #         "captions": "Gand mara"
    #     })
    file = request.files['image']
    file.save('im-received.jpg')
    # Read the image via file.stream
    return generate.runModel('im-received.jpg')

import os
import random
from flask import Flask, jsonify, request, send_file
from scipy.io.wavfile import write
from ttslib import speak

import asgiref.wsgi
import uvicorn
import numpy as np

app = Flask(__name__)


@app.route('/tts', methods=['POST', 'GET'])
def tts():
    filename = None
    try:
        data = request.args.to_dict()
        data.update(request.form.to_dict())
        sentence = data['sentence'].strip()
        speaker = data['speaker'].strip()
        if not sentence or not speaker:
            return jsonify({"status": False, "msg": "Missing parameters"})
        audio = speak(sentence, speaker)
        filename = 'output/output%08d.wav' % (random.randint(1, 99999999))
        write(filename, 22050, audio)
        return send_file(filename,
                         download_name="generated.wav",
                         conditional=False)
    finally:
        if filename is not None:
            os.remove(filename)


if __name__ == '__main__':
    # app.run(debug=True)
    asgiapp = asgiref.wsgi.WsgiToAsgi(app)
    uvicorn.run(asgiapp, host="0.0.0.0", port=8005)

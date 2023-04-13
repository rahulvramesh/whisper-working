import os
import tempfile
import flask
from flask import request, jsonify
from flask_cors import CORS
import json

from faster_whisper import WhisperModel


app = flask.Flask(__name__)
CORS(app)
model_size = "large-v2"

#HfFolder.save_token("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")

model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")



@app.route('/transcribe', methods=['POST'])
def transcribe():
    if request.method == 'POST':

        print("recieved prompt")

        # Create Temp file
        temp_dir = tempfile.mkdtemp()
        save_path: str = os.path.join(temp_dir, 'temp.wav')

        wav_file = request.files['audio_data']
        wav_file.save(save_path)

        segments, info = model.transcribe(save_path, language="en", beam_size=5, vad_filter=True)
        response_text = ""
        for segment in segments:
            response_text += segment.text
        # segment_list = [
        #     {"start": segment.start, "end": segment.end, "text": segment.text} for segment in segments
        # ]

        # Convert the list of dictionaries to a JSON string
        # json_string = json.dumps(segment_list)
        # return json_string
        data = {
            "text": response_text,
        }

        return jsonify(data), 200
    else:
        return "This endpoint only processes POST wav blob"


# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host='0.0.0.0', port=port)

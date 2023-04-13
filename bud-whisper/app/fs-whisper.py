from faster_whisper import WhisperModel
import json

model_size = "large-v2"

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("speech.mp3", beam_size=2,vad_filter=True)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))


# convert segments to json
print(segments)

# dictionary = {k: v for k, v in segments}
# json_string = json.dumps(dictionary)
# print(json_string)

# Create a list of dictionaries
segment_list = [
    {"start": segment.start, "end": segment.end, "text": segment.text} for segment in segments
]

# Convert the list of dictionaries to a JSON string
json_string = json.dumps(segment_list)

print(json_string)

for segment in segments:

    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
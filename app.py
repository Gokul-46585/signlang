
from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
from gtts import gTTS
from googletrans import Translator
import requests
import google.generativeai as genai

app = Flask(__name__)

# Load the model and labels
model = load_model("keras_model.h5", compile=False)
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Global variables
global_last_frame = None
stored_class_names = ["D", "O", "G"]
translated = ''.join(stored_class_names)

# Language code to language name mapping
def get_language_name(lang_code):
    languages = {
        "af": "afrikaans", "ar": "arabic", "bg": "bulgarian", "bn": "bengali", "bs": "bosnian",
        "ca": "catalan", "cs": "czech", "da": "danish", "de": "german", "el": "greek", "en": "english",
        "es": "spanish", "et": "estonian", "fi": "finnish", "fr": "french", "gu": "gujarati", "hi": "hindi",
        "hr": "croatian", "hu": "hungarian", "id": "indonesian", "is": "icelandic", "it": "italian",
        "iw": "hebrew", "ja": "japanese", "jw": "javanese", "km": "khmer", "kn": "kannada", "ko": "korean",
        "la": "latin", "lv": "latvian", "ml": "malayalam", "mr": "marathi", "ms": "malay", "my": "myanmar (burmese)",
        "ne": "nepali", "nl": "dutch", "no": "norwegian", "pl": "polish", "pt": "portuguese", "ro": "romanian",
        "ru": "russian", "si": "sinhala", "sk": "slovak", "sq": "albanian", "sr": "serbian", "su": "sundanese",
        "sv": "swedish", "sw": "swahili", "ta": "tamil", "te": "telugu", "th": "thai", "tl": "filipino",
        "tr": "turkish", "uk": "ukrainian", "ur": "urdu", "vi": "vietnamese", "zh": "chinese (mandarin)",
        "zh-CN": "chinese (simplified)", "zh-TW": "chinese (mandarin/taiwan)"
    }
    return languages.get(lang_code, "unknown language code")

# Predict the class of the image
def predict(image):
    size = (224, 224)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_resized = ImageOps.fit(image_pil, size, Image.LANCZOS)
    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# Save the word sound
def save_word_sound(translated, lang, file_name):
    tts = gTTS(text=translated, lang=lang)
    tts.save(file_name)

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Generate frames for normal video
def generate_frames_normal():
    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Generate frames for Mediapipe processed video
def generate_frames_mediapipe():
    global global_last_frame
    ret, test_frame = cap.read()
    if not ret:
        raise Exception("Failed to capture video frame.")
    white_bg = np.full((test_frame.shape[0], test_frame.shape[1], 3), 255, dtype=np.uint8)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        output_frame = white_bg.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    output_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
            class_name, confidence = predict(output_frame)
            cv2.putText(output_frame, f'{class_name}: {confidence:.2f}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        global_last_frame = output_frame
        ret, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Translate array to specified language
def translate_array_to_tamil(word, lang):
    global translated   
    translator = Translator()
    sound = translator.translate(word, dest=lang)
    return sound.text

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():  
    global translated 
    word = ''.join(stored_class_names)
    data = request.json
    lang = data['lang']
    translated = translate_array_to_tamil(word, lang)
    return jsonify(translated=translated)

@app.route('/video_normal')
def video_normal():
    return Response(generate_frames_normal(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_mediapipe')
def video_mediapipe():
    return Response(generate_frames_mediapipe(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/store_class_name', methods=['POST'])
def store_class_name():
    global stored_class_names
    if global_last_frame is not None:
        class_name, _ = predict(global_last_frame)
        stored_class_names.append(class_name)
        return jsonify({"message": "Class name stored", "class_name": class_name})
    return jsonify({"error": "No frame available"}), 400

@app.route('/get_stored_class_names', methods=['GET'])
def get_stored_class_names():
    return jsonify(stored_class_names)

@app.route('/clear_stored_class_names', methods=['POST'])
def clear_stored_class_names():
    global stored_class_names
    stored_class_names = []
    return jsonify({"message": "Stored class names cleared"})

@app.route('/backspace', methods=['POST'])
def backspace():
    global stored_class_names
    stored_class_names = stored_class_names[:-1]
    return jsonify({"message": "letter cleared"})

@app.route('/play_word', methods=['POST'])
def play_word():
    data = request.json
    word = ''.join(stored_class_names)
    langs = data['lang']
    sound = translate_array_to_tamil(word, langs)
    save_word_sound(sound, langs, "word_sound.mp3")
    return send_file("word_sound.mp3", as_attachment=True)

# Get report for the word
def get_report(word, langs):
    genai.configure(api_key="AIzaSyB2LBlU1icel_vqXvVgbzb2wGMpSWXaKBU")
    model = genai.GenerativeModel('gemini-pro')
    data = f"Describe the word in 2 points {word} in {langs}"
    response = model.generate_content(data)
    answer = response.text
    return answer

@app.route('/describe', methods=['GET', 'POST'])
def describe():
    search_query = stored_class_names
    data = request.json
    langs = get_language_name(data['lang'])
    word_str = ''.join(search_query)
    image_url = fetch_image(word_str)
    report = get_report(word_str, langs)
    return jsonify({"image_urls": image_url, "reports": report})

# Fetch image using Unsplash API
def fetch_image(query):
    client_id = '1nGeZVb8pXx3E2mRrZqyGiAsKfsH-oyJ_1dAprhvOWw'  
    url = "https://api.unsplash.com/search/photos"
    params = {
        'query': query,
        'client_id': client_id,
        'page': 1,
        'per_page': 1
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        return data['results'][0]['urls']['regular']
    except:
        return None

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

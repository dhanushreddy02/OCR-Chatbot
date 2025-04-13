# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, session, send_file
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pytesseract
import cohere
import torch
import os
from gtts import gTTS
import requests
from io import BytesIO
from openai import OpenAI

app = Flask(__name__)
app.secret_key = "your_secret_key"

client = OpenAI(api_key='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')

device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)


def generate_caption(image_path):
    image = Image.open(image_path)
    inputs = blip_processor(image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption


import pytesseract


def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

import cohere
import language_tool_python


co = cohere.Client("FGP7s9uFD2q3WcQ0JApJy27TPRQbErooDXWk9I32")
tool = language_tool_python.LanguageTool('en-US')

def is_grammatically_correct(text):
    matches = tool.check(text)
    return len(matches) == 0

def query_cohere(caption,ocr_text,user_question):
    if is_grammatically_correct(ocr_text):
        prompt = f"Image Caption: {caption}\nExtracted Text: {ocr_text}\nUser Question: {user_question}\nAnswer:"
    else:
        prompt = f"Image Caption: {caption}\nUser Question: {user_question}\nAnswer:"

    response = co.generate(
        model="command-xlarge",
        prompt=prompt,
        max_tokens=100,
        temperature=0.75
    )
    answer = response.generations[0].text.strip()
    return answer

@app.route("/", methods=["GET", "POST"])
def index():
    chat_history = session.get("chat_history", [])
    image_url = None # Default value for enhanced image
    audio_file = None

    if request.method == "POST":
        # Quit Button Pressed
        if "quit" in request.form:
            session.clear()  # Clears session
            return render_template("index.html", chat_history=[], image=None, audio=None)

        # Image Upload Processing
        if "image" in request.files:
            image = request.files["image"]
            if image.filename:
                image_path = os.path.join("static", image.filename)
                image.save(image_path)

                caption = generate_caption(image_path)
                ocr_text = extract_text_from_image(image_path)
                caption = caption + " " + ocr_text  # Combine caption and OCR text

                # Store Data in Session
                session["image"] = image.filename
                session["caption"] = caption
                session["ocr_text"] = ocr_text
                session["chat_history"] = []  # Reset chat history

        # Process User Question
        elif "question" in request.form:
            user_question = request.form["question"]
            answer = query_cohere(session.get("caption", ""), session.get("ocr_text", ""), user_question)

            # Convert Answer to Speech
            audio_file = text_to_speech(answer)

            # Store Conversation in Session
            chat_history.append({"user": True, "text": user_question})
            chat_history.append({"user": False, "text": answer})
            session["chat_history"] = chat_history

            return render_template(
                "index.html",
                image=session.get("image"),
                chat_history=chat_history,
                audio=audio_file,
            )

        # Generate Enhanced Image
        elif "enhance" in request.form:
            
            caption = session.get("caption", "An AI-generated image")
            response = client.images.generate(
                model="dall-e-3",
                prompt=caption,
                size="1024x1024",
                quality="standard",
                n=1,
            )

            image_url = response.data[0].url
            print(image_url)  # Print the generated image URL

            response = requests.get(image_url)
            enhanced_image = Image.open(BytesIO(response.content))
            enhanced_image.save("static/generated_image.png")  # Save in static folder

    return render_template(
        "index.html",
        image=session.get("image"),
        chat_history=session.get("chat_history", []),
        audio=session.get("audio"),
        image_url=image_url,  # Pass enhanced image URL to template
    )
def text_to_speech(text):
    if not text or text.strip() == "":
        return None  # Prevents saving an empty file

    tts = gTTS(text=text, lang="en", slow=False)
    tts.save("static/response.mp3")
    return "static/response.mp3"

    return send_file(tts_path, as_attachment=False)
      

if __name__ == "__main__":
    app.run(debug=True)

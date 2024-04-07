from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
import whisper
import queue
import threading
import torch
import numpy as np
import re
from gtts import gTTS
from openai import OpenAI
import click
import os
import time
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

r = sr.Recognizer()
try:
    with sr.Microphone() as source:
        print("Please say something")
        audio = r.listen(source)
except Exception as e:
    print(f"An error occurred: {e}")
    

@click.command()
@click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny", "base", "small", 
              "medium", "large"]))
@click.option("--english", default=False, help="Whether to use the English model", is_flag=True, type=bool)
@click.option("--energy", default=300, help="Energy level for the mic to detect", type=int)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
@click.option("--dynamic_energy", default=False, is_flag=True, help="Flag to enable dynamic energy", type=bool)
@click.option("--wake_word", default="hey computer", help="Wake word to listen for", type=str)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True, type=bool)

def main(model, english, energy, pause, dynamic_energy, wake_word, verbose):
    if model != "large" and english:
        model = model + ".en"

    audio_model = whisper.load_model(model)
    audio_queue = queue.Queue()
    result_queue = queue.Queue()

    threading.Thread(target=record_audio, args=(audio_queue, energy, pause, dynamic_energy,)).start()
    threading.Thread(target=transcribe_forever, args=(audio_queue, result_queue, audio_model, english, wake_word, 
                     verbose,)).start()
    threading.Thread(target=reply, args=(result_queue, verbose,)).start()

    time.sleep(500)

    while True:
        print(result_queue.get())

def record_audio(audio_queue, energy, pause, dynamic_energy):
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = energy
    recognizer.pause_threshold = pause
    recognizer.dynamic_energy_threshold = dynamic_energy 
    
    with sr.Microphone(sample_rate=16000) as source:
        print("Listening...")
        i = 0
        while True:
            audio = recognizer.listen(source)
            torch_audio = torch.from_numpy(
                np.frombuffer(audio.get_raw_data(), np.int16)
                .flatten()
                .astype(np.float32) / 32768.0
            )

            audio_data = torch_audio
            audio_queue.put_nowait(audio_data)
            i += 1
            print("recorded the " + str(i) + "th audio")

def transcribe_forever(audio_queue, result_queue, audio_model, english, 
           wake_word, verbose):
    i = 0
    while True:
        audio_data = audio_queue.get()
        if english:
            result = audio_model.transcribe(audio_data, language='english')
        else:
            result = audio_model.transcribe(audio_data)
        predicted_text = result["text"]
        if predicted_text.strip().lower().startswith(wake_word.strip().lower()):
            pattern = re.compile(re.escape(wake_word), re.IGNORECASE)
            predicted_text = pattern.sub("", predicted_text).strip()
            punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            predicted_text = predicted_text.translate({ord(i): 
                    None for i in punc})

            if verbose:
                print("You said the wake word.. Processing {}...".
                       format(predicted_text))

            result_queue.put_nowait(predicted_text)
        else:
            if verbose:
                print("You did not say the wake word.. Ignoring")
        print("transcribed the " + str(i) + "th audio, the audio content is : " + predicted_text)
        i += 1

def reply(result_queue, verbose):
    i = 0
    while True:
        question = result_queue.get()
        prompt = "Q: {}?\nA:".format(question)
        
        data = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=0.5,
            max_tokens=100,
            n=1,
            stop=["\n"]
        )
        try:
            answer = data["choices"][0]["text"]
            mp3_obj = gTTS(text=answer, lang="en", slow=False)
        except Exception as e:
            choices = [
                "I'm sorry, I don't know the answer to that",
                "I'm not sure I understand",
                "I'm not sure I can answer that",
                "Please repeat the question in a different way"
            ]
            mp3_obj = gTTS(text=choices[np.random.randint(0, len(choices))], 
                           lang="en", slow=False)
            if verbose:
                print(e)

        mp3_obj.save("reply.mp3")
        reply_audio = AudioSegment.from_mp3("reply.mp3")
        play(reply_audio)
        print("replied the " + str(i) + "th audio, the answer content is " + answer)
        i += 1

main()
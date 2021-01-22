from __future__ import division, print_function
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import sounddevice as sd
from scipy.io.wavfile import write
import pandas as pd
import librosa.display
import librosa
import noisereduce as nr
import random
import pyaudio
import pickle 
import speech_recognition 
# Flask utils
from flask import Flask, redirect, url_for, request, render_template

r=speech_recognition.Recognizer()

# Define a flask app
app = Flask(__name__)
loaded_model = pickle.load(open("RFmodel", 'rb')) 


@app.route('/addrec',methods = ['POST', 'GET'])
def addrec():
	if request.method == 'POST':
            # fs = 44100  
            # seconds = 10  # Duration of recording

            # myrecording = sd.rec((seconds * fs), samplerate=fs, channels=2)
            # render_template('index.html', predict_value='Recording started()......')
            # sd.wait()  # Wait until recording is finished
            # render_template('index.html', predict_value='Recording Finished......')
            # render_template('index.html', predict_value='Wait Predicting......')
            # write('output.wav', fs, myrecording)
            # sound,sr=librosa.load("output.wav")
            # dsound,sr=librosa.load('dnoise.wav')
            # noise_reduced = nr.reduce_noise(audio_clip=sound, noise_clip=dsound, prop_decrease=1.0)
            # write('output.wav', sr, noise_reduced)
            # r=sr.Recognizer()

            with speech_recognition.Microphone() as source:
                audio=r.record(source, duration=10)
                with open('output.wav','wb') as f:
                    f.write(audio.get_wav_data())
            clip, sample_rate = librosa.load("output.wav", sr=None)
            if(np.max(clip)<0.03):
                return render_template('index.html', predict_value='Result : No Music')
            else:
                test=[]
                signal,sr=librosa.load("output.wav")
                mfccs = librosa.feature.mfcc(signal, sr=sr)
                for j in mfccs:
                    test.append(j.mean()) 
                test=np.array(test)
                test=np.reshape(test,(1,-1))
                predictions = loaded_model.predict(test)
                if(predictions[0]==0):
                    return render_template('index.html', predict_value='Result : Music File')
                if(predictions[0]==1):
                    return render_template('index.html', predict_value='Result :  Noise File')

            # except:
            #     return render_template('index.html', predict_value='Somthing went wrong please Try again')

                
    
	return render_template("index.html",predict_value='Please upload file')


@app.route('/', methods=['GET','POST'])
def index():
  
        return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        try:
        # Get the file from post request
           
            sound = AudioSegment.from_mp3(f)
            sound.export("music.wav", format="wav")
            clip, sample_rate = librosa.load("music.wav", sr=None)
            if(np.max(clip)<0.01):
                return render_template('index.html', predict_value='Result : No Music')
            else:
                test=[]
                signal,sr=librosa.load("music.wav")
                mfccs = librosa.feature.mfcc(signal, sr=sr)
                for j in mfccs:
                    test.append(j.mean()) 
                test=np.array(test)
                test=np.reshape(test,(1,-1))
                predictions = loaded_model.predict(test)
                if(predictions[0]==0):
                    return render_template('index.html', predict_value='Result : Music File')
                if(predictions[0]==1):
                    return render_template('index.html', predict_value='Result :  Noise File')
        except:

            return render_template('index.html', predict_value='Please Upload mp3 file...')



    return render_template("index.html")


if __name__ == '__main__':
     app.run()
    #app.run(host='0.0.0.0',port=8080)
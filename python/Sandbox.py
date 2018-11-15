# Sandbox.py
# Ashish D'Souza
# November 14th, 2018

import autocomplete
import gtts
import os


gtts.gTTS("Good morning", lang="en").save("speak.mp3")
os.system("mpg123 speak.mp3; rm speak.mp3")
import os
from pydub import AudioSegment
import filetype

fileUrl = "jazz.00038.wav"

kind = filetype.guess(fileUrl)

if kind != None:
    if kind.mime == "audio/mpeg":
        print("This is mp3 file.")
    elif kind.mime == "audio/x-wav":
        print("This is wav file.")

from pydub import AudioSegment

src = "music1.mp3"
dst = "test.wav"

audSeg = AudioSegment.from_mp3(src)
audSeg.export(dst, format="wav")

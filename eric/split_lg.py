from vad import VoiceActivityDetector

filename = '/home/leferrae/Desktop/These/Kunwok/tour_nabagardi/20190723_085214.wav'
v = VoiceActivityDetector(filename)
v.plot_detected_speech_regions()

from pychorus import find_and_output_chorus

audio_file = "data/raw/test.wav"
output_file = "data/chorus/test_chorus.wav"

chorus_start = find_and_output_chorus(audio_file, output_file)

print("Chorus start time:", chorus_start)
print("Chorus saved to:", output_file)
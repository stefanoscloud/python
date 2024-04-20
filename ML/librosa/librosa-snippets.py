# librosa Python library

# librosa.load(‘audio.wav’, sr=None, mono=True) 
# Librosa.display.waveshow(y= samples, sr=sampling_rate) – Generate a waveform of the audio file. 
# stft_mat= librosa.stft(samples) – Return a matrix of sample data that has had a short-term Fourier transform applied.
# Db= librosa.amplitude_to_db(np.abs(sftf_mat_), ref= np.max) – Convert the default amplitude values to a relative decibel scale
# Librosa.display.specshow(db, y_axis = ‘log’, x_axis = ‘time’, sr = sampling_rate) – Generate a spectrogram from the db-coverted sample values, using a logarithmic scale and the native sampling rate.
# Soundfile.write(‘processed_audio.wav’, samples, samplerate= sampling_rate) – Save the sample data as an audio file.

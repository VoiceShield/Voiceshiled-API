import io
from pydub import AudioSegment

def save_byte_data_as_wav(byte_data, wav_file_path):
    # Create a BytesIO object from the byte data
    byte_io = io.BytesIO(byte_data)

    # Convert the byte data to an AudioSegment
    audio = AudioSegment.from_file(byte_io, format="raw", frame_rate=44100, channels=2, sample_width=2)

    # Export the AudioSegment to a WAV file
    audio.export(wav_file_path, format="wav")

# Example usage:
byte_data = b'\x00\x01\x02\x03'  # Replace with your actual byte data
wav_file_path = 'output.wav'

save_byte_data_as_wav(byte_data, wav_file_path)

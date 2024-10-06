import speech_recognition as sr
from pydub import AudioSegment
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def transcribe_audio_chunk(audio_chunk, recognizer, language):
    try:
        with sr.AudioFile(audio_chunk) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data, language=language), None
    except sr.UnknownValueError:
        return None, "Speech recognition could not understand audio"
    except sr.RequestError as e:
        return None, f"Could not request results from Google Speech Recognition service: {e}"

def format_time(seconds):
    """Convert seconds to hh:mm:ss format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def determine_chunk_length(audio_length_seconds):
    """Determine chunk length dynamically based on the total audio length."""
    if audio_length_seconds < 10 * 60:  # less than 10 minutes
        return 10  # 10-second chunks
    elif 10 * 60 <= audio_length_seconds < 60 * 60:  # 10 minutes to 1 hour
        return 30  # 30-second chunks
    else:  # more than 1 hour
        return 60  # 60-second chunks (can be increased if needed)

def search_phrase_in_audio(wav_file, search_phrase, language):
    audio = AudioSegment.from_wav(wav_file)

    # Get the total audio length in seconds
    audio_length_ms = len(audio)
    audio_length_seconds = audio_length_ms / 1000

    # Dynamically determine chunk length based on audio length
    chunk_length = determine_chunk_length(audio_length_seconds)
    chunk_length_ms = chunk_length * 1000  # Convert seconds to milliseconds

    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Divide the audio into larger chunks (e.g., 60 seconds)
    audio_length_ms = len(audio)
    chunk_length_ms = chunk_length * 1000  # Convert seconds to milliseconds
    num_chunks = math.ceil(audio_length_ms / chunk_length_ms)

    phrase_positions = []
    processed_chunks = 0

    def process_chunk(i):
        start_time = i * chunk_length_ms
        end_time = min((i + 1) * chunk_length_ms, audio_length_ms)
        chunk = audio[start_time:end_time]

        # Save chunk as temporary WAV file for transcription
        chunk_wav_file = f"chunk_{i}.wav"
        chunk.export(chunk_wav_file, format="wav")

        # Transcribe the chunk
        transcription, error = transcribe_audio_chunk(chunk_wav_file, recognizer, language=language)

        # Clean up chunk file immediately after processing
        os.remove(chunk_wav_file)

        if error:
            print(f"Error in chunk {i}: {error}")
            return None

        # Search for the phrase in the transcription
        if transcription:
            transcription = transcription.lower()
            search_phrase_lower = search_phrase.lower()

            if search_phrase_lower in transcription:
                phrase_start_idx = transcription.find(search_phrase_lower)
                word_ratio = phrase_start_idx / len(transcription)
                phrase_time = start_time + word_ratio * (end_time - start_time)
                return phrase_time / 1000  # Return time in seconds

        return None

    # Use ThreadPoolExecutor to process chunks in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, i) for i in range(num_chunks)]

        for future in as_completed(futures):
            result = future.result()
            processed_chunks += 1

            # Progress monitoring
            print(f"Processed {processed_chunks}/{num_chunks} chunks.")

            if result is not None:
                phrase_positions.append(result)

    # Display results
    if phrase_positions:
        print(f"The phrase '{search_phrase}' was found at the following times:")
        for pos in phrase_positions:
            print(format_time(pos))  # Output in hh:mm:ss format
    else:
        print(f"The phrase '{search_phrase}' was not found in the audio.")


# Sample usage
wav_file = "audio/videoplayback.wav"
search_phrase = "ещё раз"
search_phrase_in_audio(wav_file, search_phrase, "ru-RU")
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")
# Or suppress specific warnings, such as UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import necessary libraries
import re
import torchaudio
import librosa
torchaudio.set_audio_backend("soundfile")
import whisper
from transformers import HubertForCTC, Wav2Vec2Processor, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from google.cloud import texttospeech
from jiwer import wer  # Library for Word Error Rate (WER) calculation

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")
# Or suppress specific warnings, such as UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set device to GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Whisper and Hugging Face Model Initialization
Whisper_model = whisper.load_model("base").to(device)
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
hubert_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(device)

# Use a grammar correction model designed for minimal adjustments, if available
correction_model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1").to(device)
correction_tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")

# Google Text-To-Speech Initialization
client = texttospeech.TextToSpeechClient()

# Helper Function: Load and preprocess audio


import torch

def load_audio(path):
    audio_input, sample_rate = librosa.load(path, sr=16000)  # Force 16kHz for HuBERT
    audio_tensor = torch.tensor(audio_input).unsqueeze(0)
    return audio_tensor, sample_rate


# Step 1: Transcribe Audio using Whisper
def transcribe_whisper(audio_path):
    result = Whisper_model.transcribe(audio_path)
    return result["text"]

# Step 2: Transcribe Audio using HuBERT
def transcribe_hubert(audio_path):
    audio_input, sample_rate = load_audio(audio_path)
    if sample_rate != 16000:
        audio_input = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio_input)
    inputs = processor(audio_input.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = hubert_model(input_values=inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Step 3: Error Correction with Grammar Correction Model
def correct_text(text):
    # Split the text into manageable chunks if it's too long
    chunk_size = 512  # You can adjust this based on model constraints
    corrected_text = ""
    
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        inputs = correction_tokenizer(chunk, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        # Generate the corrected output with a controlled length
        summary_ids = correction_model.generate(
            inputs.input_ids, 
            max_length=512,   # Ensure output length doesn't exceed 512 tokens
            num_beams=4, 
            early_stopping=True, 
            no_repeat_ngram_size=2,   # Prevent repetition of n-grams
            top_p=1,           # Limit potential words to the top 95% of probabilities
            temperature=0.7       # Control randomness of output
        )
        
        # Decode the generated output into human-readable text
        corrected_text += correction_tokenizer.decode(summary_ids[0], skip_special_tokens=True) + " "
    
    return corrected_text.strip()

# Step 4: Convert text To Speech using Google Text-to-Speech
def text_to_speech(text, output_filename="output.mp3"):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    with open(output_filename, "wb") as out:
        out.write(response.audio_content)
        print(f"Audio content written to the file {output_filename}")

# Accuracy Evaluation using Word Error Rate (WER)
def calculate_accuracy(ground_truth, transcribed_text):
    error_rate = wer(ground_truth, transcribed_text)
    accuracy_percentage = (1 - error_rate) * 100
    # Normalize both strings: lowercase and remove punctuation
    ground_truth_normalized = re.sub(r'[^\w\s]', '', ground_truth.lower())
    transcribed_text_normalized = re.sub(r'[^\w\s]', '', transcribed_text.lower())
    error_rate = wer(ground_truth_normalized, transcribed_text_normalized)
    accuracy_percentage = (1 - error_rate) * 100
    return round(accuracy_percentage, 2)

# Main process
def process_audio_file(audio_path, ground_truth_text):
    print("THE ORIGINAL TRANSCRIPTION: ", ground_truth_text)
    
    # Transcribe with Whisper
    transcription_whisper = transcribe_whisper(audio_path)
    print("Whisper Transcription: ", transcription_whisper)

    # Transcribe with HuBERT
    transcription_hubert = transcribe_hubert(audio_path)
    print("HuBERT Transcription: ", transcription_hubert)

    # Correct Transcriptions
    corrected_text_whisper = correct_text(transcription_whisper)
    corrected_text_hubert = correct_text(transcription_hubert)
    print("Corrected Whisper Transcription: ", corrected_text_whisper)
    print("Corrected HuBERT Transcription: ", corrected_text_hubert)

    # Calculate and Print Accuracy
    accuracy_whisper = calculate_accuracy(ground_truth_text, corrected_text_whisper)
    accuracy_hubert = calculate_accuracy(ground_truth_text, corrected_text_hubert)
    print(f"Whisper Transcription Accuracy: {accuracy_whisper}%")
    print(f"HuBERT Transcription Accuracy: {accuracy_hubert}%")

    # Text-To-Speech Conversion
    text_to_speech(corrected_text_whisper, output_filename="whisper_output.mp3")
    text_to_speech(corrected_text_hubert, output_filename="hubert_output.mp3")

# Example call for one audio file with ground truth text
process_audio_file(
    r" ", #path to the audio file locally stored
    ground_truth_text=" " #the ground truth added to compare with the transcription
)

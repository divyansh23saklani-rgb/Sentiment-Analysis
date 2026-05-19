import os
import argparse
from pydub import AudioSegment
import numpy as np
import torch
import librosa
import soundfile as sf

def convert_ogg_to_wav(ogg_file_path):
    """Convert OGG file to WAV format as required by the voice cloning model."""
    print(f"Converting {ogg_file_path} to WAV format...")
    
    wav_file_path = ogg_file_path.replace('.ogg', '.wav')
    
    # Load the OGG file
    audio = AudioSegment.from_ogg(ogg_file_path)
    
    # Export as WAV (ensure 16kHz sample rate for compatibility)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(wav_file_path, format="wav")
    
    print(f"Converted file saved as {wav_file_path}")
    return wav_file_path

def clone_voice_and_speak(reference_audio_path, text_to_speak, output_path):
    """Clone a voice using pre-trained models and synthesize new speech."""
    try:
        # Import required libraries inside function to handle import errors gracefully
        import torch
        from tortoise.api import TextToSpeech
        from tortoise.utils.audio import load_audio, load_voice
        
        print("Loading Tortoise TTS voice cloning model...")
        
        # Initialize Tortoise TTS
        tts = TextToSpeech()
        
        # Check if we need to convert OGG to WAV
        if reference_audio_path.lower().endswith('.ogg'):
            reference_audio_path = convert_ogg_to_wav(reference_audio_path)
        
        print(f"Loading reference voice from {reference_audio_path}")
        
        # Fixed line: properly handle the return value from load_audio
        voice_samples = load_audio(reference_audio_path, 22050)
        voice_samples = torch.tensor(voice_samples).unsqueeze(0)
        
        # Generate speech with the cloned voice
        print(f"Generating speech: '{text_to_speak}'")
        gen_audio = tts.tts(text_to_speak, voice_samples=voice_samples, 
                           conditioning_latents=None, preset="standard")
        
        # Convert from torch tensor and save the audio
        audio_data = gen_audio.squeeze(0).cpu().numpy()
        sf.write(output_path, audio_data, 24000)
        
        print(f"Generated speech saved to {output_path}")
        
        # Play the audio
        play_audio(output_path)
        
        return output_path
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nIt looks like you need to install the Tortoise TTS library.")
        print("Please install it using the following commands:")
        print("\npip install tortoise-tts")
        print("pip install transformers")
        print("pip install librosa")
        print("pip install soundfile")
        
        # Fallback to simpler method if Tortoise isn't available
        print("\nFalling back to simpler voice clone method...")
        return simple_voice_cloning(reference_audio_path, text_to_speak, output_path)

def simple_voice_cloning(reference_audio_path, text_to_speak, output_path):
    """Simplified voice cloning using pyttsx3 for systems without GPU."""
    try:
        import pyttsx3
        
        print("Using pyttsx3 for speech synthesis (basic TTS without voice cloning)...")
        
        # Initialize the TTS engine
        engine = pyttsx3.init()
        
        # Get voice characteristics from the reference audio
        # (For pyttsx3 we can only adjust rate, volume, and voice selection)
        try:
            # Try to analyze the reference audio for basic properties
            if reference_audio_path.lower().endswith('.ogg'):
                reference_audio_path = convert_ogg_to_wav(reference_audio_path)
                
            # Load audio and get basic properties
            y, sr = librosa.load(reference_audio_path)
            
            # Extract pitch information to estimate speech rate
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 150
            
            # Set rate based on pitch (very simple approximation)
            # Lower pitch usually means slower speech
            rate_factor = avg_pitch / 150  # 150 Hz is considered average
            engine.setProperty('rate', int(175 * rate_factor))  # Default is 200
            
            # Set volume
            engine.setProperty('volume', 0.8)  # 0.0 to 1.0
            
            # Try to pick an appropriate voice
            voices = engine.getProperty('voices')
            if len(voices) > 1:
                # Simple gender estimation based on pitch
                if avg_pitch > 170:  # Higher pitch typically indicates female voice
                    for voice in voices:
                        if "female" in voice.name.lower():
                            engine.setProperty('voice', voice.id)
                            break
                else:  # Lower pitch typically indicates male voice
                    for voice in voices:
                        if "male" in voice.name.lower():
                            engine.setProperty('voice', voice.id)
                            break
        except Exception as e:
            print(f"Couldn't analyze reference audio: {e}")
            print("Using default voice settings")
        
        # Generate speech
        print(f"Generating speech: '{text_to_speak}'")
        engine.save_to_file(text_to_speak, output_path)
        engine.runAndWait()
        
        print(f"Generated speech saved to {output_path}")
        
        # Play the audio
        play_audio(output_path)
        
        return output_path
    
    except ImportError:
        print("\nError: Could not import pyttsx3.")
        print("Please install it using: pip install pyttsx3")
        print("For now, generating a text file with the speech content.")
        
        # As a last resort, just save the text
        with open(output_path.replace('.wav', '.txt'), 'w') as f:
            f.write(text_to_speak)
        
        print(f"Text saved to {output_path.replace('.wav', '.txt')}")
        return None

def play_audio(audio_path):
    """Play the generated audio using system's default player."""
    try:
        import platform
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            os.system(f"afplay '{audio_path}'")
        elif system == 'Linux':
            os.system(f"aplay '{audio_path}'")
        elif system == 'Windows':
            os.system(f"start '{audio_path}'")
        
        print("Playing audio output...")
    except Exception as e:
        print(f"Couldn't play audio automatically: {e}")
        print(f"Please play {audio_path} manually to hear the result.")

def main():
    parser = argparse.ArgumentParser(description="Clone a voice from reference audio and generate speech")
    parser.add_argument("--reference", required=True, help="Path to reference audio file (OGG or WAV)")
    parser.add_argument("--text", required=True, help="Text to synthesize with the cloned voice")
    parser.add_argument("--output", default="output_speech.wav", help="Output speech file path")
    
    args = parser.parse_args()
    
    print("\n===== Voice Cloning System =====")
    print(f"Reference audio: {args.reference}")
    print(f"Text to speak: {args.text}")
    print(f"Output path: {args.output}")
    print("===============================\n")
    
    # Clone voice and generate speech
    clone_voice_and_speak(args.reference, args.text, args.output)

if __name__ == "__main__":
    main()
import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Output text file
output_file = "live_transcription.txt"

print("🎤 Say something! (Press Ctrl+C to stop)")

# Use the microphone for real-time input
with sr.Microphone() as source:
    # Adjust for ambient noise
    recognizer.adjust_for_ambient_noise(source)
    
    # Open file to write
    with open(output_file, "w", encoding="utf-8") as f:
        try:
            while True:
                # Listen for audio
                audio = recognizer.listen(source)

                try:
                    # Recognize speech using Google Web Speech API
                    text = recognizer.recognize_google(audio, language="en-IN")
                    print("📝 You said:", text)
                    
                    # Write recognized text to file
                    f.write(text + "\n")
                    f.flush()  # Ensure data is written immediately
                except sr.UnknownValueError:
                    print("🤷‍♂️ Could not understand audio")
                except sr.RequestError as e:
                    print(f"❌ Could not request results; {e}")
        except KeyboardInterrupt:
            print("\n👋 Exiting and saving the transcription...")

print(f"✅ Transcription saved to {output_file}")

import pyttsx3
import threading

class TextToSpeech:
    def __init__(self):
        # Initialize the text-to-speech engine
        self.engine = pyttsx3.init()
        
        # Set properties
        self.engine.setProperty('rate', 150)    # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        
        # Get available voices and set to a female voice if available
        voices = self.engine.getProperty('voices')
        if len(voices) > 1:  # If there's more than one voice
            self.engine.setProperty('voice', voices[1].id)  # Index 1 is usually female voice
            
        # Create a lock for thread safety
        self.lock = threading.Lock()
        
    def speak(self, text):
        """
        Speak the given text using text-to-speech
        
        Args:
            text (str): Text to be spoken
        """
        # Run in a separate thread to avoid blocking the main program
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()
        
    def _speak_thread(self, text):
        """
        Internal method to handle speech in a separate thread
        
        Args:
            text (str): Text to be spoken
        """
        with self.lock:
            self.engine.say(text)
            self.engine.runAndWait() 
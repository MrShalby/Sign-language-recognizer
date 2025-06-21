import cv2
import mediapipe as mp
import numpy as np
from hand_detector import HandDetector
from sign_classifier import SignClassifier
from text_to_speech import TextToSpeech
from collections import deque
import time

class SignLanguageApp:
    def __init__(self):
        # Initialize camera with higher resolution
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)# for width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # for height
        
        # Initialize components
        self.hand_detector = HandDetector(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.sign_classifier = SignClassifier()
        self.tts_engine = TextToSpeech()
        
        # Initialize UI and tracking variables
        self.prev_prediction = ""
        self.prediction_time = time.time()
        self.letter_buffer = []
        self.word_buffer = []
        self.prediction_history = deque(maxlen=10)
        self.prediction_counts = {}
        self.is_recording = True
        self.space_pressed = False
        
        # Settings
        self.min_prediction_time = 0.5
        self.letter_timeout = 2.0
        self.confidence_threshold = 0.8
        self.required_stable_predictions = 3
        self.roi_size = 400
        
        # Load ASL reference images
        self.reference_image = self.load_reference_image()
        
    def load_reference_image(self):
        """Create a reference image showing ASL alphabet"""
        # Create a black image
        img = np.zeros((200, 800, 3), dtype=np.uint8)
        # Add letters A-Z
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        x, y = 10, 30
        for i, letter in enumerate(letters):
            if i > 0 and i % 13 == 0:  # New row every 13 letters
                x = 10
                y += 80
            cv2.putText(img, f"{letter}", (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            x += 60
        return img

    def run(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to capture frame")
                break
                
            frame = cv2.flip(frame, 1)
            roi_frame = self.create_roi(frame)
            roi_frame, hand_landmarks, is_stable = self.hand_detector.detect_hands(roi_frame)
            self.update_roi(frame, roi_frame)
            
            current_prediction = "Unknown"
            if hand_landmarks and is_stable:
                current_prediction = self.sign_classifier.classify_gesture(hand_landmarks[0])
                self.update_prediction_counts(current_prediction)
                
                if self.is_prediction_stable(current_prediction):
                    if (current_prediction != self.prev_prediction and 
                        time.time() - self.prediction_time > self.min_prediction_time):
                        self.handle_new_prediction(current_prediction)
                        self.prediction_time = time.time()
                        self.prev_prediction = current_prediction
            else:
                self.prediction_counts.clear()
            
            self.draw_enhanced_ui(frame, current_prediction)
            
            if not self.handle_keyboard_input():
                break
            
        self.cap.release()
        cv2.destroyAllWindows()

    def create_roi(self, frame):
        """Create a region of interest in the center of the frame"""
        h, w = frame.shape[:2]
        roi_x = w//2 - self.roi_size//2
        roi_y = h//2 - self.roi_size//2
        
        # Extract ROI
        roi = frame[roi_y:roi_y+self.roi_size, roi_x:roi_x+self.roi_size].copy()
        
        # Draw ROI boundary
        cv2.rectangle(frame, (roi_x, roi_y), 
                     (roi_x+self.roi_size, roi_y+self.roi_size),
                     (0, 255, 0), 2)
        
        return roi

    def update_roi(self, frame, roi_frame):
        """Update the ROI in the main frame"""
        h, w = frame.shape[:2]
        roi_x = w//2 - self.roi_size//2
        roi_y = h//2 - self.roi_size//2
        frame[roi_y:roi_y+self.roi_size, roi_x:roi_x+self.roi_size] = roi_frame

    def update_prediction_counts(self, prediction):
        """Update the counts of consecutive predictions"""
        # Reset counts if this is a new prediction
        if prediction not in self.prediction_counts:
            self.prediction_counts.clear()
        
        # Update count for current prediction
        self.prediction_counts[prediction] = self.prediction_counts.get(prediction, 0) + 1
        
    def is_prediction_stable(self, prediction):
        """Check if the prediction is stable"""
        return (prediction in self.prediction_counts and 
                self.prediction_counts[prediction] >= self.required_stable_predictions)

    def handle_new_prediction(self, prediction):
        """Handle new letter prediction"""
        if prediction != "Unknown":
            # Add to prediction history
            self.prediction_history.append(prediction)
            print(f"Detected Sign: {prediction}")
            
            # Add to letter buffer if recording
            if self.is_recording:
                self.letter_buffer.append(prediction)
                
            # Speak the letter
            self.tts_engine.speak(prediction)

    def add_space(self):
        """Add space between words"""
        if self.letter_buffer:
            word = ''.join(self.letter_buffer)
            self.word_buffer.append(word)
            self.letter_buffer = []
            # Speak the word
            self.tts_engine.speak(word)

    def clear_buffers(self):
        """Clear all buffers"""
        self.letter_buffer = []
        self.word_buffer = []
        self.prediction_history.clear()
        self.prediction_counts.clear()

    def handle_keyboard_input(self):
        """Handle keyboard input and return False if should quit"""
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        elif key == ord('c'):
            self.clear_buffers()
        elif key == ord(' '):
            if not self.space_pressed:
                self.add_space()
                self.space_pressed = True
        else:
            self.space_pressed = False
        return True

    def draw_enhanced_ui(self, frame, current_prediction):
        """Draw enhanced UI with better visualization"""
        h, w = frame.shape[:2]
        
        # Create info panel on the right
        panel_width = 300
        info_panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
        
        # Draw current detection status
        cv2.putText(info_panel, "Detection Status:", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show current letter being detected
        status_color = (0, 255, 0) if current_prediction != "Unknown" else (0, 0, 255)
        cv2.putText(info_panel, f"Current: {current_prediction}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Show stability percentage
        if self.prediction_counts:
            max_prediction = max(self.prediction_counts.items(), key=lambda x: x[1])
            stability = min(100, int(max_prediction[1] / self.required_stable_predictions * 100))
            cv2.putText(info_panel, f"Stability: {stability}%", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw word formation section
        cv2.putText(info_panel, "Word Formation:", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show current word being formed
        current_word = ''.join(self.letter_buffer)
        cv2.putText(info_panel, f"Letters: {current_word}", (10, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show complete sentence
        cv2.putText(info_panel, "Sentence:", (10, 230),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        sentence = ' '.join(self.word_buffer + ([current_word] if current_word else []))
        # Split sentence into multiple lines if needed
        words = sentence.split()
        line = ""
        y_offset = 270
        for word in words:
            if len(line + " " + word) < 20:  # Adjust based on panel width
                line = line + " " + word if line else word
            else:
                cv2.putText(info_panel, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += 30
                line = word
        if line:
            cv2.putText(info_panel, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw instructions
        y_offset = h - 200
        instructions = [
            "Instructions:",
            "1. Show hand sign in green box",
            "2. Hold until stable (green)",
            "3. Press SPACE for word break",
            "4. Press 'C' to clear",
            "5. Press 'Q' to quit"
        ]
        for instruction in instructions:
            cv2.putText(info_panel, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_offset += 30
        
        # Add reference alphabet at the bottom of the main frame
        ref_height = 200
        frame[h-ref_height:h, 0:800] = self.reference_image
        
        # Combine main frame with info panel
        combined_frame = np.hstack([frame, info_panel])
        
        # Display the combined frame
        cv2.imshow("Sign Language Detector", combined_frame)

if __name__ == "__main__":
    app = SignLanguageApp()
    app.run() 
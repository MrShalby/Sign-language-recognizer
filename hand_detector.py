import mediapipe as mp
import cv2
import numpy as np

class HandDetector:
    def __init__(self, static_mode=False, max_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8):
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            model_complexity=1  # Use more accurate model
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        # Initialize tracking variables
        self.prev_landmarks = None
        self.smoothing_factor = 0.5
        self.stable_frames = 0
        self.min_stable_frames = 3

    def detect_hands(self, frame):
        """
        Detect hand landmarks in the given frame with improved stability
        
        Args:
            frame: Input image frame
            
        Returns:
            frame: Processed frame with landmarks drawn
            landmarks: List of detected hand landmarks
            is_stable: Boolean indicating if hand position is stable
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = self.hands.process(frame_rgb)
        landmarks = []
        is_stable = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Apply smoothing to landmarks
                smoothed_landmarks = self._smooth_landmarks(hand_landmarks)
                
                # Draw enhanced landmarks and connections
                self._draw_enhanced_landmarks(frame, smoothed_landmarks)
                
                # Check stability
                is_stable = self._check_stability(smoothed_landmarks)
                
                # Store smoothed landmarks
                landmarks.append(smoothed_landmarks)
                self.prev_landmarks = smoothed_landmarks

            # Draw hand status
            status_color = (0, 255, 0) if is_stable else (0, 0, 255)
            cv2.putText(frame, "Hand Stable" if is_stable else "Stabilizing...",
                       (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, status_color, 2)
        else:
            self.prev_landmarks = None
            self.stable_frames = 0

        return frame, landmarks, is_stable

    def _smooth_landmarks(self, current_landmarks):
        """Apply exponential smoothing to landmarks for stability"""
        if self.prev_landmarks is None:
            return current_landmarks

        smoothed = self.mp_hands.HandLandmark(0)
        for i, landmark in enumerate(current_landmarks.landmark):
            if i < len(self.prev_landmarks.landmark):
                prev = self.prev_landmarks.landmark[i]
                landmark.x = self.smoothing_factor * landmark.x + (1 - self.smoothing_factor) * prev.x
                landmark.y = self.smoothing_factor * landmark.y + (1 - self.smoothing_factor) * prev.y
                landmark.z = self.smoothing_factor * landmark.z + (1 - self.smoothing_factor) * prev.z

        return current_landmarks

    def _check_stability(self, landmarks):
        """Check if hand position is stable"""
        if self.prev_landmarks is None:
            self.stable_frames = 0
            return False

        # Calculate movement between frames
        total_movement = 0
        for i, landmark in enumerate(landmarks.landmark):
            if i < len(self.prev_landmarks.landmark):
                prev = self.prev_landmarks.landmark[i]
                movement = np.sqrt((landmark.x - prev.x)**2 + 
                                 (landmark.y - prev.y)**2 + 
                                 (landmark.z - prev.z)**2)
                total_movement += movement

        # Check if movement is below threshold
        if total_movement < 0.1:  # Adjust threshold as needed
            self.stable_frames += 1
        else:
            self.stable_frames = 0

        return self.stable_frames >= self.min_stable_frames

    def _draw_enhanced_landmarks(self, frame, hand_landmarks):
        """Draw enhanced landmarks with better visibility"""
        # Draw landmarks with custom styling
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            self.mp_draw.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )

        # Draw landmark numbers for better debugging
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                       0.3, (255, 255, 255), 1) 
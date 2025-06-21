import numpy as np
import cv2

class SignClassifier:
    def __init__(self):
        # Initialize finger landmark indices
        self.finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        self.finger_bases = [2, 5, 9, 13, 17]  # Corresponding finger bases
        self.finger_mids = [3, 7, 11, 15, 19]  # Middle points of fingers

    def classify_gesture(self, landmarks):
        """
        Classify the hand gesture based on landmark positions
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            str: Predicted gesture label
        """
        # Extract landmark coordinates
        points = []
        for landmark in landmarks.landmark:
            points.append([landmark.x, landmark.y, landmark.z])
        points = np.array(points)

        # Calculate key features
        finger_states = self._get_finger_states(points)
        angles = self._calculate_finger_angles(points)
        distances = self._calculate_finger_distances(points)
        
        # Classify gestures based on features
        return self._determine_letter(finger_states, angles, distances, points)

    def _get_finger_states(self, points):
        """Determine if each finger is extended or closed"""
        states = []
        
        # Special case for thumb
        thumb_angle = self._calculate_angle(points[0], points[2], points[4])
        states.append(thumb_angle > 150)
        
        # For other fingers
        for tip, mid, base in zip(self.finger_tips[1:], self.finger_mids[1:], self.finger_bases[1:]):
            finger_extended = points[tip][1] < points[mid][1] < points[base][1]
            states.append(finger_extended)
            
        return states

    def _calculate_finger_angles(self, points):
        """Calculate angles between fingers"""
        angles = []
        for i in range(len(self.finger_tips)-1):
            v1 = points[self.finger_tips[i]] - points[self.finger_bases[i]]
            v2 = points[self.finger_tips[i+1]] - points[self.finger_bases[i+1]]
            angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / 
                (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
            angles.append(angle)
        return angles

    def _calculate_finger_distances(self, points):
        """Calculate distances between fingertips"""
        distances = []
        for i in range(len(self.finger_tips)):
            for j in range(i+1, len(self.finger_tips)):
                dist = np.linalg.norm(points[self.finger_tips[i]] - points[self.finger_tips[j]])
                distances.append(dist)
        return distances

    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        v1 = p1 - p2
        v2 = p3 - p2
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / 
            (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))

    def _determine_letter(self, finger_states, angles, distances, points):
        """
        Determine ASL letter based on finger states and angles
        Returns: ASL letter or 'Unknown'
        """
        # Unpack finger states for easier reference
        thumb_up, index_up, middle_up, ring_up, pinky_up = finger_states
        
        # Helper function to check if fingers are close
        def are_fingers_close(finger1, finger2, threshold=0.1):
            return np.linalg.norm(points[self.finger_tips[finger1]] - 
                                points[self.finger_tips[finger2]]) < threshold

        # A-Z ASL Recognition logic
        if not any([index_up, middle_up, ring_up, pinky_up]) and thumb_up:
            return 'A'
        
        elif all([index_up, middle_up, ring_up, pinky_up]) and not thumb_up:
            return 'B'
        
        elif all([not x for x in [index_up, middle_up, ring_up, pinky_up]]) and \
             self._is_hand_curved(points):
            return 'C'
        
        elif index_up and not any([middle_up, ring_up, pinky_up]):
            return 'D'
        
        elif all([not x for x in [index_up, middle_up, ring_up, pinky_up]]) and \
             self._is_thumb_across_palm(points):
            return 'E'
        
        elif not any([middle_up, ring_up, pinky_up]) and index_up and \
             self._is_thumb_across_index(points):
            return 'F'
        
        elif index_up and thumb_up and \
             all(points[self.finger_tips[i]][1] > points[self.finger_bases[i]][1] 
                 for i in [2, 3, 4]):
            return 'G'
        
        elif index_up and middle_up and not any([ring_up, pinky_up]):
            return 'H'
        
        elif pinky_up and not any([index_up, middle_up, ring_up]):
            return 'I'
        
        elif pinky_up and index_up and middle_up and \
             self._is_moving_hand(points):  # Note: J is actually a motion
            return 'J'
        
        elif index_up and middle_up and thumb_up and \
             points[self.finger_tips[0]][0] < points[self.finger_bases[1]][0]:
            return 'K'
        
        elif index_up and thumb_up and \
             self._is_l_shape(points):
            return 'L'
        
        elif index_up and middle_up and ring_up and \
             all(are_fingers_close(i, i+1) for i in range(1, 3)):
            return 'M'
        
        elif index_up and middle_up and \
             are_fingers_close(1, 2):
            return 'N'
        
        elif all([not x for x in [index_up, middle_up, ring_up, pinky_up]]) and \
             self._is_o_shape(points):
            return 'O'
        
        elif index_up and middle_up and ring_up and \
             self._are_fingers_together(points, [1, 2, 3]):
            return 'P'
        
        elif self._is_q_shape(points):
            return 'Q'
        
        elif index_up and middle_up and \
             self._are_fingers_crossed(points, 1, 2):
            return 'R'
        
        elif all([not x for x in [middle_up, ring_up, pinky_up]]) and \
             self._is_fist_shape(points):
            return 'S'
        
        elif index_up and self._is_thumb_between_fingers(points):
            return 'T'
        
        elif index_up and middle_up and \
             self._is_u_shape(points):
            return 'U'
        
        elif index_up and middle_up and \
             self._is_v_shape(points):
            return 'V'
        
        elif index_up and middle_up and ring_up and \
             self._is_w_shape(points):
            return 'W'
        
        elif self._is_x_shape(points):
            return 'X'
        
        elif thumb_up and pinky_up and \
             self._is_y_shape(points):
            return 'Y'
        
        elif index_up and \
             self._is_z_motion(points):  # Note: Z is actually a motion
            return 'Z'

        return "Unknown"

    def _is_hand_curved(self, points):
        """Check if hand is in a C shape"""
        return self._calculate_angle(points[4], points[8], points[20]) < 90

    def _is_thumb_across_palm(self, points):
        """Check if thumb is across palm (for E)"""
        return points[4][0] < points[5][0]

    def _is_thumb_across_index(self, points):
        """Check if thumb crosses index finger (for F)"""
        return np.linalg.norm(points[4] - points[8]) < 0.1

    def _is_moving_hand(self, points):
        """Placeholder for J motion detection"""
        return False  # Would need temporal information for actual implementation

    def _is_l_shape(self, points):
        """Check if hand forms L shape"""
        angle = self._calculate_angle(points[4], points[0], points[8])
        return 80 < angle < 100

    def _is_o_shape(self, points):
        """Check if fingers form O shape"""
        return np.linalg.norm(points[4] - points[8]) < 0.15

    def _are_fingers_together(self, points, finger_indices):
        """Check if specified fingers are together"""
        return all(np.linalg.norm(points[self.finger_tips[i]] - 
                                points[self.finger_tips[i+1]]) < 0.1 
                  for i in finger_indices[:-1])

    def _is_q_shape(self, points):
        """Check if hand forms Q shape"""
        return (self._is_o_shape(points) and 
                points[self.finger_tips[0]][1] < points[self.finger_bases[0]][1])

    def _are_fingers_crossed(self, points, finger1, finger2):
        """Check if two fingers are crossed"""
        return (points[self.finger_tips[finger1]][0] > 
                points[self.finger_tips[finger2]][0])

    def _is_fist_shape(self, points):
        """Check if hand is in fist shape"""
        return all(points[tip][1] > points[base][1] 
                  for tip, base in zip(self.finger_tips, self.finger_bases))

    def _is_thumb_between_fingers(self, points):
        """Check if thumb is between fingers (for T)"""
        return (points[4][1] > points[5][1] and 
                points[4][1] < points[8][1])

    def _is_u_shape(self, points):
        """Check if fingers form U shape"""
        return (np.linalg.norm(points[8] - points[12]) < 0.1 and 
                points[8][1] < points[5][1])

    def _is_v_shape(self, points):
        """Check if fingers form V shape"""
        angle = self._calculate_angle(points[8], points[5], points[12])
        return 20 < angle < 60

    def _is_w_shape(self, points):
        """Check if fingers form W shape"""
        angles = [self._calculate_angle(points[8], points[5], points[12]),
                 self._calculate_angle(points[12], points[9], points[16])]
        return all(20 < angle < 60 for angle in angles)

    def _is_x_shape(self, points):
        """Check if hand forms X shape"""
        return (points[8][1] > points[7][1] and 
                points[8][1] < points[6][1])

    def _is_y_shape(self, points):
        """Check if hand forms Y shape"""
        thumb_pinky_angle = self._calculate_angle(points[4], points[0], points[20])
        return 40 < thumb_pinky_angle < 100

    def _is_z_motion(self, points):
        """Placeholder for Z motion detection"""
        return False  # Would need temporal information for actual implementation 
"""
Perspective correction and cropping - FIXED
"""

import cv2
import numpy as np

class PerspectiveCorrector:
    
    @staticmethod
    def apply_perspective_transform(image, corners):
        """Apply perspective transform to get bird's eye view"""
        
        if corners is None or len(corners) != 4:
            print("Invalid corners for perspective transform")
            return image
        
        # Convert corners to float32
        src_corners = np.array(corners, dtype=np.float32)
        
        # Calculate width and height of the document
        width_top = np.linalg.norm(src_corners[1] - src_corners[0])
        width_bottom = np.linalg.norm(src_corners[2] - src_corners[3])
        max_width = max(int(width_top), int(width_bottom))
        
        height_left = np.linalg.norm(src_corners[3] - src_corners[0])
        height_right = np.linalg.norm(src_corners[2] - src_corners[1])
        max_height = max(int(height_left), int(height_right))
        
        # Ensure minimum dimensions
        max_width = max(max_width, 400)
        max_height = max(max_height, 500)
        
        # Destination corners (bird's eye view)
        dst_corners = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
        
        # Apply perspective transform
        warped = cv2.warpPerspective(
            image,
            matrix,
            (max_width, max_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        # Remove any remaining borders
        warped = PerspectiveCorrector.remove_borders(warped)
        
        return warped
    
    @staticmethod
    def remove_borders(image):
        """Remove black or white borders from the warped image"""
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Find non-white pixels (threshold at 250)
        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours of non-white area
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of largest contour
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            
            # Add small margin
            margin = 5
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.shape[1] - x, w + 2 * margin)
            h = min(image.shape[0] - y, h + 2 * margin)
            
            # Crop
            if w > 50 and h > 50:
                return image[y:y+h, x:x+w]
        
        return image
    
    @staticmethod
    def rotate_image(image, angle):
        """Rotate image by given angle"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (width, height))
        return rotated
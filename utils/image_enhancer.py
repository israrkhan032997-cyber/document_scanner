"""
Image quality enhancement - Produces clear output
"""

import cv2
import numpy as np

class ImageEnhancer:
    
    @staticmethod
    def enhance_document(image, mode="color"):
        """Enhance the scanned document"""
        
        if image is None or image.size == 0:
            return image
        
        try:
            if mode == "bw":
                # Black and white
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Adaptive threshold for clean B&W
                enhanced = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
                return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            elif mode == "grayscale":
                # Grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            else:
                # Color mode - enhance contrast
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_enhanced = clahe.apply(l)
                
                enhanced_lab = cv2.merge([l_enhanced, a, b])
                enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                
                # Slight brightness adjustment
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
                
                return enhanced
                
        except Exception as e:
            print(f"Enhancement error: {e}")
            return image
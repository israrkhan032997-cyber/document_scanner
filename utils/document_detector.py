"""
Document boundary detection - WITH MULTIPLE FALLBACK METHODS
"""

import cv2
import numpy as np

class DocumentDetector:
    
    # ============ EDGE DETECTION MODULE ============
    @staticmethod
    def edge_detection(image):
        """Enhanced edge detection using multiple methods"""
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply bilateral filter to preserve edges while reducing noise
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Method 1: Canny with adaptive thresholds
        median = np.median(denoised)
        lower = int(max(0, 0.66 * median))
        upper = int(min(255, 1.33 * median))
        edges_canny = cv2.Canny(denoised, lower, upper)
        
        # Method 2: Morphological gradient for edge detection
        kernel = np.ones((3, 3), np.uint8)
        gradient = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT, kernel)
        
        # Method 3: Difference of Gaussian
        blur1 = cv2.GaussianBlur(denoised, (3, 3), 0)
        blur2 = cv2.GaussianBlur(denoised, (9, 9), 0)
        dog = blur1 - blur2
        
        # Combine all edge detection methods
        edges_combined = cv2.addWeighted(edges_canny, 0.5, gradient, 0.3, 0)
        edges_combined = cv2.addWeighted(edges_combined, 0.8, dog, 0.2, 0)
        
        # Enhance edges
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges_combined, kernel, iterations=2)
        edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)
        
        return edges_closed
    
    # ============ CONTOUR DETECTION WITH MULTIPLE STRATEGIES ============
    @staticmethod
    def find_contours_strategy_1(edge_image, original_image):
        """Strategy 1: Find largest quadrilateral contour"""
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            image_area = original_image.shape[0] * original_image.shape[1]
            
            # Skip if too small or too large
            if area < 0.02 * image_area or area > 0.98 * image_area:
                continue
            
            # Approximate contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            if len(approx) == 4:
                return approx, contour
            
            # Try with different epsilon values
            for epsilon in [0.03, 0.04, 0.05]:
                approx2 = cv2.approxPolyDP(contour, epsilon * perimeter, True)
                if len(approx2) == 4:
                    return approx2, contour
        
        return None, None
    
    @staticmethod
    def find_contours_strategy_2(original_image):
        """Strategy 2: Use color segmentation for document detection"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
        
        # Extract different channels
        h, s, v = cv2.split(hsv)
        l, a, b = cv2.split(lab)
        
        # Try different thresholding methods
        methods = []
        
        # Saturation channel (documents usually have low saturation)
        _, sat_thresh = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)
        methods.append(sat_thresh)
        
        # Value channel
        _, val_thresh = cv2.threshold(v, 100, 255, cv2.THRESH_BINARY)
        methods.append(val_thresh)
        
        # Lightness channel
        _, light_thresh = cv2.threshold(l, 120, 255, cv2.THRESH_BINARY)
        methods.append(light_thresh)
        
        # Adaptive threshold on grayscale
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        methods.append(adaptive)
        
        # Try each method
        for method in methods:
            # Clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            cleaned = cv2.morphologyEx(method, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Sort by area
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    image_area = original_image.shape[0] * original_image.shape[1]
                    
                    if area < 0.05 * image_area or area > 0.95 * image_area:
                        continue
                    
                    # Approximate
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                    
                    if len(approx) == 4:
                        return approx, contour
                    
                    # Try to get bounding rectangle
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    return box, contour
        
        return None, None
    
    @staticmethod
    def find_contours_strategy_3(original_image):
        """Strategy 3: Use morphological operations to find document"""
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Otsu threshold
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to connect document edges
        kernel = np.ones((15, 15), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Sort by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                image_area = original_image.shape[0] * original_image.shape[1]
                
                if area < 0.1 * image_area:
                    continue
                
                # Get bounding rectangle
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                
                # Approximate
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                if len(approx) == 4:
                    return approx, contour
                else:
                    return box, contour
        
        return None, None
    
    @staticmethod
    def find_contours_strategy_4(original_image):
        """Strategy 4: Use edge detection with Hough lines"""
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None and len(lines) > 4:
            # Create a blank image to draw lines
            line_image = np.zeros_like(gray)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
            
            # Dilate lines
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(line_image, kernel, iterations=2)
            
            # Find contours of line intersections
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour
                largest = max(contours, key=cv2.contourArea)
                
                # Get bounding rectangle
                rect = cv2.minAreaRect(largest)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                return box, largest
        
        return None, None
    
    # ============ CORNER ORDERING MODULE ============
    @staticmethod
    def order_corners(corners):
        """Order corners in correct sequence"""
        if corners is None or len(corners) != 4:
            return corners
        
        pts = corners.reshape(4, 2).astype(np.float32)
        
        # Calculate center
        center = np.mean(pts, axis=0)
        
        # Calculate angles from center
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        
        # Sort by angle
        sorted_indices = np.argsort(angles)
        ordered = pts[sorted_indices]
        
        # Ensure top-left is first
        # Find point with smallest y (top)
        min_y_idx = np.argmin(ordered[:, 1])
        top_points = ordered[ordered[:, 1] == ordered[min_y_idx, 1]]
        
        if len(top_points) >= 2:
            # Among top points, leftmost is top-left
            left_idx = np.argmin(top_points[:, 0])
            top_left = top_points[left_idx]
            
            # Find index of top-left in ordered array
            start_idx = np.where((ordered == top_left).all(axis=1))[0][0]
            ordered = np.roll(ordered, -start_idx, axis=0)
        
        return ordered
    
    # ============ VALIDATE CORNERS ============
    @staticmethod
    def validate_corners(corners, image_shape):
        """Validate if corners form a valid document"""
        if corners is None or len(corners) != 4:
            return False
        
        height, width = image_shape[:2]
        
        # Check if corners are within image bounds (with generous margin)
        margin = 100
        for corner in corners:
            x, y = corner
            if x < -margin or x > width + margin or y < -margin or y > height + margin:
                return False
        
        # Calculate areas and check if reasonable
        # Use shoelace formula
        area = 0
        for i in range(4):
            x1, y1 = corners[i]
            x2, y2 = corners[(i+1) % 4]
            area += (x1 * y2 - x2 * y1)
        area = abs(area) / 2
        
        image_area = height * width
        if area < 0.05 * image_area or area > 1.1 * image_area:
            return False
        
        return True
    
    # ============ MAIN DETECTION PIPELINE ============
    @staticmethod
    def detect_document(image):
        """Complete document detection with multiple fallback strategies"""
        
        original = image.copy()
        height, width = original.shape[:2]
        
        print(f"Image size: {width} x {height}")
        
        # Try Strategy 1: Edge detection + Contour detection
        print("Attempting Strategy 1: Edge detection...")
        edges = DocumentDetector.edge_detection(original)
        corners, contour = DocumentDetector.find_contours_strategy_1(edges, original)
        
        if corners is not None:
            print("✓ Strategy 1 succeeded!")
            ordered_corners = DocumentDetector.order_corners(corners)
            if DocumentDetector.validate_corners(ordered_corners, original.shape):
                return ordered_corners, contour, edges
        
        # Try Strategy 2: Color segmentation
        print("Attempting Strategy 2: Color segmentation...")
        corners, contour = DocumentDetector.find_contours_strategy_2(original)
        
        if corners is not None:
            print("✓ Strategy 2 succeeded!")
            ordered_corners = DocumentDetector.order_corners(corners)
            if DocumentDetector.validate_corners(ordered_corners, original.shape):
                return ordered_corners, contour, edges
        
        # Try Strategy 3: Morphological operations
        print("Attempting Strategy 3: Morphological operations...")
        corners, contour = DocumentDetector.find_contours_strategy_3(original)
        
        if corners is not None:
            print("✓ Strategy 3 succeeded!")
            ordered_corners = DocumentDetector.order_corners(corners)
            if DocumentDetector.validate_corners(ordered_corners, original.shape):
                return ordered_corners, contour, edges
        
        # Try Strategy 4: Hough lines
        print("Attempting Strategy 4: Hough lines...")
        corners, contour = DocumentDetector.find_contours_strategy_4(original)
        
        if corners is not None:
            print("✓ Strategy 4 succeeded!")
            ordered_corners = DocumentDetector.order_corners(corners)
            if DocumentDetector.validate_corners(ordered_corners, original.shape):
                return ordered_corners, contour, edges
        
        # Final fallback: Use entire image as document
        print("All strategies failed. Using full image as document...")
        height, width = original.shape[:2]
        fallback_corners = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        return fallback_corners, None, edges
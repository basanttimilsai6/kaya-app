import cv2
import numpy as np
import json
from scipy import ndimage
from skimage import measure, morphology, feature
from skimage.filters import gaussian, sobel
from skimage.segmentation import watershed
from skimage.feature import local_binary_pattern
import warnings
warnings.filterwarnings('ignore')

class SkinAnalyzerJSON:
    def __init__(self):
        # Initialize face detector
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def detect_face(self, image):
        """Detect face in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def get_skin_mask(self, image):
        """Create a mask for skin regions using color-based segmentation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        lower_hsv = np.array([0, 20, 70])
        upper_hsv = np.array([20, 255, 255])
        lower_ycrcb = np.array([0, 135, 85])
        upper_ycrcb = np.array([255, 180, 135])
        
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        return skin_mask
    
    def count_spots_and_acne(self, image, face_region):
        """Count spots and acne"""
        x, y, w, h = face_region
        face_img = image[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_face, (3, 3), 0)
        
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY_INV, 11, 2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        spots_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 200:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.3:
                        spots_count += 1
        
        return spots_count
    
    def analyze_redness(self, image, face_region):
        """Analyze redness percentage"""
        x, y, w, h = face_region
        face_img = image[y:y+h, x:x+w]
        skin_mask = self.get_skin_mask(face_img)
        hsv_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        hue = hsv_face[:, :, 0]
        saturation = hsv_face[:, :, 1]
        
        red_mask1 = ((hue >= 0) & (hue <= 10)) | ((hue >= 160) & (hue <= 180))
        red_mask2 = saturation > 40
        redness_mask = red_mask1 & red_mask2 & (skin_mask > 0)
        
        skin_pixels = np.sum(skin_mask > 0)
        red_pixels = np.sum(redness_mask)
        redness_percentage = (red_pixels / skin_pixels * 100) if skin_pixels > 0 else 0
        
        return round(redness_percentage, 2)
    
    def count_eye_issues(self, image, face_region):
        """Count eye bags and dark circles"""
        x, y, w, h = face_region
        face_img = image[y:y+h, x:x+w]
        eyes = self.eye_detector.detectMultiScale(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), 1.1, 5)
        
        eye_bags_count = 0
        dark_circles_count = 0
        
        if len(eyes) > 0:
            for (ex, ey, ew, eh) in eyes:
                eye_bag_region_y_start = ey + eh
                eye_bag_region_y_end = min(ey + eh + int(eh * 0.8), face_img.shape[0])
                eye_bag_region_x_start = max(0, ex - int(ew * 0.2))
                eye_bag_region_x_end = min(ex + ew + int(ew * 0.2), face_img.shape[1])
                
                if eye_bag_region_y_end > eye_bag_region_y_start:
                    eye_region = face_img[eye_bag_region_y_start:eye_bag_region_y_end, 
                                        eye_bag_region_x_start:eye_bag_region_x_end]
                    
                    if eye_region.size > 0:
                        bags, circles = self._count_eye_region_issues(eye_region)
                        eye_bags_count += bags
                        dark_circles_count += circles
        
        return eye_bags_count, dark_circles_count
    
    def _count_eye_region_issues(self, eye_region):
        """Count issues in eye region"""
        if eye_region.size == 0:
            return 0, 0
            
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_eye, (5, 5), 0)
        
        # Eye bags detection
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.abs(sobel_y)
        threshold = np.mean(gradient_magnitude) + 0.8 * np.std(gradient_magnitude)
        eye_bag_candidates = gradient_magnitude > threshold
        
        # Dark circles detection
        mean_intensity = np.mean(gray_eye)
        dark_threshold = mean_intensity - 0.5 * np.std(gray_eye)
        dark_circle_candidates = gray_eye < dark_threshold
        
        # Count eye bags
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        eye_bag_mask = cv2.morphologyEx(eye_bag_candidates.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        contours_bags, _ = cv2.findContours(eye_bag_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        eye_bags_count = len([c for c in contours_bags if cv2.contourArea(c) > 30])
        
        # Count dark circles
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark_circle_mask = cv2.morphologyEx(dark_circle_candidates.astype(np.uint8), cv2.MORPH_CLOSE, kernel2)
        contours_circles, _ = cv2.findContours(dark_circle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dark_circles_count = len([c for c in contours_circles if cv2.contourArea(c) > 50])
        
        return eye_bags_count, dark_circles_count
    
    def count_wrinkles(self, image, face_region):
        """Count wrinkles and calculate density"""
        x, y, w, h = face_region
        face_img = image[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_face, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
        
        wrinkle_count = 0
        total_length = 0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > 15:
                    wrinkle_count += 1
                    total_length += length
        
        face_area = w * h
        wrinkle_density = round(total_length / face_area, 4) if face_area > 0 else 0
        
        return wrinkle_count, wrinkle_density
    
    def analyze_oiliness(self, image, face_region):
        """Analyze skin oiliness"""
        x, y, w, h = face_region
        face_img = image[y:y+h, x:x+w]
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        skin_mask = self.get_skin_mask(face_img)
        
        brightness = hsv[:, :, 2]
        skin_brightness = brightness[skin_mask > 0]
        
        if len(skin_brightness) > 0:
            oiliness_threshold = np.percentile(skin_brightness, 85)
            oily_pixels = np.sum(skin_brightness > oiliness_threshold)
            total_skin_pixels = len(skin_brightness)
            oiliness_percentage = (oily_pixels / total_skin_pixels * 100)
        else:
            oiliness_percentage = 0
        
        if oiliness_percentage > 25:
            oiliness_level = "High"
        elif oiliness_percentage > 15:
            oiliness_level = "Medium"
        elif oiliness_percentage > 5:
            oiliness_level = "Low"
        else:
            oiliness_level = "Very Low"
        
        return round(oiliness_percentage, 2), oiliness_level
    
    def analyze_moisture(self, image, face_region):
        """Analyze skin moisture"""
        x, y, w, h = face_region
        face_img = image[y:y+h, x:x+w]
        skin_mask = self.get_skin_mask(face_img)
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray_face.astype(np.float32), -1, kernel)
        local_sqr_mean = cv2.filter2D((gray_face.astype(np.float32))**2, -1, kernel)
        local_variance = local_sqr_mean - local_mean**2
        texture_variation = np.sqrt(np.maximum(local_variance, 0))
        
        skin_texture = texture_variation[skin_mask > 0]
        avg_texture_variation = np.mean(skin_texture) if len(skin_texture) > 0 else 0
        moisture_score = max(0, 100 - (avg_texture_variation * 2))
        
        if moisture_score > 80:
            moisture_level = "Very High"
        elif moisture_score > 60:
            moisture_level = "High"
        elif moisture_score > 40:
            moisture_level = "Medium"
        elif moisture_score > 20:
            moisture_level = "Low"
        else:
            moisture_level = "Very Low"
        
        return round(moisture_score, 2), moisture_level
    
    def analyze_firmness(self, image, face_region):
        """Analyze skin firmness"""
        x, y, w, h = face_region
        face_img = image[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_face, (3, 3), 0)
        
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        avg_gradient = np.mean(gradient_magnitude)
        firmness_score = min(100, avg_gradient * 2)
        
        if firmness_score > 80:
            firmness_level = "Very High"
        elif firmness_score > 60:
            firmness_level = "High"
        elif firmness_score > 40:
            firmness_level = "Medium"
        elif firmness_score > 20:
            firmness_level = "Low"
        else:
            firmness_level = "Very Low"
        
        return round(firmness_score, 2), firmness_level
    
    def analyze_radiance(self, image, face_region):
        """Analyze skin radiance"""
        x, y, w, h = face_region
        face_img = image[y:y+h, x:x+w]
        skin_mask = self.get_skin_mask(face_img)
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        skin_lightness = l_channel[skin_mask > 0]
        
        if len(skin_lightness) > 0:
            avg_lightness = np.mean(skin_lightness)
            lightness_std = np.std(skin_lightness)
            radiance_score = (avg_lightness / 255 * 100) - (lightness_std / 255 * 50)
            radiance_score = max(0, min(100, radiance_score))
        else:
            radiance_score = 0
        
        if radiance_score > 80:
            radiance_level = "Very High"
        elif radiance_score > 60:
            radiance_level = "High"
        elif radiance_score > 40:
            radiance_level = "Medium"
        elif radiance_score > 20:
            radiance_level = "Low"
        else:
            radiance_level = "Very Low"
        
        return round(radiance_score, 2), radiance_level
    
    def count_pores(self, image, face_region):
        """Count visible pores"""
        x, y, w, h = face_region
        face_img = image[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_face, (3, 3), 0)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
        _, pore_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(pore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pores_count = len([c for c in contours if 1 < cv2.contourArea(c) < 20])
        
        face_area = w * h
        pore_density = round(pores_count / face_area * 10000, 2) if face_area > 0 else 0
        
        return pores_count, pore_density
    
    def analyze_skin_texture(self, image, face_region):
        """Analyze overall skin texture"""
        x, y, w, h = face_region
        face_img = image[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        skin_mask = self.get_skin_mask(face_img)
        
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_face, n_points, radius, method='uniform')
        
        skin_lbp = lbp[skin_mask > 0]
        if len(skin_lbp) > 0:
            texture_variance = np.var(skin_lbp)
            texture_score = max(0, 100 - (texture_variance / 10))
        else:
            texture_score = 0
        
        if texture_score > 80:
            texture_quality = "Very Smooth"
        elif texture_score > 60:
            texture_quality = "Smooth"
        elif texture_score > 40:
            texture_quality = "Moderate"
        elif texture_score > 20:
            texture_quality = "Rough"
        else:
            texture_quality = "Very Rough"
        
        return round(texture_score, 2), texture_quality
    
    def analyze_image(self, image_path):
        """Perform comprehensive skin analysis and return JSON"""
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Could not load image {image_path}"}
        
        faces = self.detect_face(image)
        if len(faces) == 0:
            return {"error": "No faces detected in the image"}
        
        # Analyze the first detected face
        face = faces[0]
        
        # Get all analysis results
        spots_count = self.count_spots_and_acne(image, face)
        redness_percentage = self.analyze_redness(image, face)
        eye_bags_count, dark_circles_count = self.count_eye_issues(image, face)
        wrinkle_count, wrinkle_density = self.count_wrinkles(image, face)
        oiliness_percentage, oiliness_level = self.analyze_oiliness(image, face)
        moisture_score, moisture_level = self.analyze_moisture(image, face)
        firmness_score, firmness_level = self.analyze_firmness(image, face)
        radiance_score, radiance_level = self.analyze_radiance(image, face)
        pores_count, pore_density = self.count_pores(image, face)
        texture_score, texture_quality = self.analyze_skin_texture(image, face)
        
        # Create JSON response with f-string values
        analysis_result = {
            "status": "success",
            "analysis": {
                "spots": {
                    "count": f"{spots_count}",
                    "severity": f"{'High' if spots_count > 10 else 'Medium' if spots_count > 5 else 'Low'}"
                },
                "acne": {
                    "count": f"{spots_count}",
                    "severity": f"{'High' if spots_count > 10 else 'Medium' if spots_count > 5 else 'Low'}"
                },
                "redness": {
                    "percentage": f"{redness_percentage}",
                    "level": f"{'High' if redness_percentage > 15 else 'Medium' if redness_percentage > 8 else 'Low'}"
                },
                "eye_area": {
                    "eye_bags_count": f"{eye_bags_count}",
                    "dark_circles_count": f"{dark_circles_count}"
                },
                "wrinkles": {
                    "count": f"{wrinkle_count}",
                    "density": f"{wrinkle_density}",
                    "severity": f"{'High' if wrinkle_count > 15 else 'Medium' if wrinkle_count > 8 else 'Low'}"
                },
                "oiliness": {
                    "percentage": f"{oiliness_percentage}",
                    "level": f"{oiliness_level}"
                },
                "moisture": {
                    "score": f"{moisture_score}",
                    "level": f"{moisture_level}"
                },
                "firmness": {
                    "score": f"{firmness_score}",
                    "level": f"{firmness_level}"
                },
                "radiance": {
                    "score": f"{radiance_score}",
                    "level": f"{radiance_level}"
                },
                "pores": {
                    "count": f"{pores_count}",
                    "density": f"{pore_density}",
                    "visibility": f"{'High' if pores_count > 50 else 'Medium' if pores_count > 25 else 'Low'}"
                },
                "skin_texture": {
                    "score": f"{texture_score}",
                    "quality": f"{texture_quality}"
                }
            },
            "summary": {
                "overall_skin_health": f"{self._calculate_overall_health(spots_count, redness_percentage, wrinkle_count, moisture_score, firmness_score, radiance_score, texture_score)}",
                "main_concerns": [f"{concern}" for concern in self._identify_main_concerns(
                    spots_count, redness_percentage, eye_bags_count, 
                    dark_circles_count, wrinkle_count, oiliness_percentage, 
                    moisture_score, pores_count
                )],
                "recommendations": [f"{rec}" for rec in self._generate_recommendations(
                    spots_count, redness_percentage, oiliness_percentage, 
                    moisture_score, wrinkle_count, radiance_score
                )]
            }
        }

        return analysis_result

    def _calculate_overall_health(self, spots, redness, wrinkles, moisture, firmness, radiance, texture):
        """Calculate overall skin health score"""
        # Penalty for problems
        problem_score = max(0, 100 - (spots * 2) - (redness * 2) - (wrinkles * 1.5))
        
        # Positive factors
        quality_score = (moisture + firmness + radiance + texture) / 4
        
        # Combined score
        overall_score = (problem_score + quality_score) / 2
        
        if overall_score >= 80:
            return {"score": round(overall_score, 1), "rating": "Excellent"}
        elif overall_score >= 65:
            return {"score": round(overall_score, 1), "rating": "Good"}
        elif overall_score >= 50:
            return {"score": round(overall_score, 1), "rating": "Fair"}
        elif overall_score >= 35:
            return {"score": round(overall_score, 1), "rating": "Poor"}
        else:
            return {"score": round(overall_score, 1), "rating": "Very Poor"}
    
    def _identify_main_concerns(self, spots, redness, eye_bags, dark_circles, wrinkles, oiliness, moisture, pores):
        """Identify main skin concerns"""
        concerns = []
        
        if spots > 8:
            concerns.append("Acne/Spots")
        if redness > 12:
            concerns.append("Redness/Irritation")
        if eye_bags > 2:
            concerns.append("Eye Bags")
        if dark_circles > 2:
            concerns.append("Dark Circles")
        if wrinkles > 12:
            concerns.append("Fine Lines/Wrinkles")
        if oiliness > 25:
            concerns.append("Excessive Oiliness")
        if moisture < 40:
            concerns.append("Dry Skin")
        if pores > 40:
            concerns.append("Enlarged Pores")
        
        return concerns if concerns else ["No major concerns detected"]
    
    def _generate_recommendations(self, spots, redness, oiliness, moisture, wrinkles, radiance):
        """Generate skincare recommendations"""
        recommendations = []
        
        if spots > 5:
            recommendations.append("Use products with salicylic acid or benzoyl peroxide for acne control")
        if redness > 10:
            recommendations.append("Consider gentle, anti-inflammatory skincare products")
        if oiliness > 20:
            recommendations.append("Use oil-free, mattifying products and clay masks")
        if moisture < 50:
            recommendations.append("Increase moisturizing routine with hyaluronic acid products")
        if wrinkles > 10:
            recommendations.append("Consider anti-aging products with retinoids or peptides")
        if radiance < 50:
            recommendations.append("Use vitamin C serums and exfoliating products for brightness")
        
        # Always include basic recommendations
        recommendations.extend([
            "Maintain daily sunscreen use (SPF 30+)",
            "Follow consistent cleansing routine twice daily",
            "Stay hydrated and maintain healthy diet"
        ])
        
        return recommendations

# Usage functions
def analyze_skin_to_json(image_path):
    """Simple function to get JSON analysis of skin"""
    print(image_path)
    analyzer = SkinAnalyzerJSON()
    data = analyzer.analyze_image(image_path)
    return data

# def save_analysis_to_file(image_path, output_file="skin_analysis.json"):
#     """Analyze skin and save results to JSON file"""
#     result = analyze_skin_to_json(image_path)
    
#     with open(output_file, 'w') as f:
#         json.dump(result, f, indent=2)
    
#     print(f"Analysis saved to {output_file}")
#     return result
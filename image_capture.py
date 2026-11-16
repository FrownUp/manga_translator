# image_capture.py
import cv2
import numpy as np
from PIL import Image, ImageGrab
import base64
from io import BytesIO
from typing import Dict, List, Tuple, Optional
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests

class ImageCapture:
    """Capture images from web pages using multiple methods"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def capture_from_url(self, url: str) -> Optional[np.ndarray]:
        """Capture image from direct URL"""
        try:
            # Download image from URL
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img_array = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return img
        except Exception as e:
            self.logger.warning(f"URL capture failed: {e}")
        return None
    
    def capture_visible_region(self, driver, element=None) -> np.ndarray:
        """Capture visible region of webpage or specific element"""
        try:
            if element:
                # Capture specific element
                location = element.location
                size = element.size
                left = location['x']
                top = location['y']
                right = left + size['width']
                bottom = top + size['height']
                
                screenshot = driver.get_screenshot_as_png()
                img = Image.open(BytesIO(screenshot))
                img = img.crop((left, top, right, bottom))
            else:
                # Capture entire viewport
                screenshot = driver.get_screenshot_as_png()
                img = Image.open(BytesIO(screenshot))
            
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.logger.error(f"Region capture failed: {e}")
            return None
    
    def capture_from_canvas(self, driver, canvas_selector: str) -> Optional[np.ndarray]:
        """Capture image from HTML5 canvas element"""
        try:
            # Extract canvas data via JavaScript
            script = """
            var canvas = document.querySelector(arguments[0]);
            if (canvas) {
                return canvas.toDataURL('image/png').split(',')[1];
            }
            return null;
            """
            canvas_data = driver.execute_script(script, canvas_selector)
            
            if canvas_data:
                img_data = base64.b64decode(canvas_data)
                img_array = np.frombuffer(img_data, np.uint8)
                return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            self.logger.warning(f"Canvas capture failed: {e}")
        return None
    
    def smart_capture(self, driver, target_element=None) -> Dict:
        """Smart image capture with fallback methods"""
        result = {
            'image': None,
            'method': 'unknown',
            'element_info': {}
        }
        
        # Try direct URL first if it's an img element
        if target_element and target_element.tag_name == 'img':
            src = target_element.get_attribute('src')
            if src and src.startswith(('http', '//')):
                img = self.capture_from_url(src)
                if img is not None:
                    result['image'] = img
                    result['method'] = 'direct_url'
                    result['element_info'] = {
                        'tag': 'img',
                        'src': src,
                        'naturalWidth': target_element.get_attribute('naturalWidth'),
                        'naturalHeight': target_element.get_attribute('naturalHeight')
                    }
                    return result
        
        # Try canvas extraction
        if target_element and target_element.tag_name == 'canvas':
            img = self.capture_from_canvas(driver, f'canvas#{target_element.get_attribute("id")}')
            if img is not None:
                result['image'] = img
                result['method'] = 'canvas'
                return result
        
        # Fallback to screenshot
        img = self.capture_visible_region(driver, target_element)
        if img is not None:
            result['image'] = img
            result['method'] = 'screenshot'
            return result
        
        return result

    def image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        try:
            success, buffer = cv2.imencode('.png', image)
            if not success:
                raise ValueError("Could not encode image to PNG")
            
            img_str = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            self.logger.error(f"Image to base64 conversion failed: {e}")
            raise

    def base64_to_image(self, base64_str: str) -> np.ndarray:
        """Convert base64 string to numpy image"""
        try:
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            
            img_data = base64.b64decode(base64_str)
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Could not decode base64 image data")
                
            return img
        except Exception as e:
            self.logger.error(f"Base64 to image conversion failed: {e}")
            raise

    def capture_with_selenium(self, url: str, element_selector: str = None, method: str = 'screenshot') -> Dict:
        """
        High-level capture method using Selenium
        
        Args:
            url: URL to capture from
            element_selector: CSS selector for specific element
            method: Capture method ('screenshot', 'canvas', 'smart')
            
        Returns:
            Dictionary with capture results
        """
        try:
            # Use webdriver-manager for automatic driver management
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            
            # Setup Chrome options
            chrome_options = Options()
            chrome_options.add_argument('--headless')  # Run in background
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--disable-extensions')
            
            # Initialize driver with webdriver-manager
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            try:
                driver.get(url)
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # Additional wait for page to fully load
                WebDriverWait(driver, 10).until(
                    lambda driver: driver.execute_script("return document.readyState") == "complete"
                )
                
                target_element = None
                if element_selector:
                    target_element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, element_selector))
                    )
                
                # Perform capture based on method
                if method == 'canvas' and element_selector:
                    result = self.capture_from_canvas(driver, element_selector)
                    if result is not None:
                        return {
                            'image': result,
                            'method': 'canvas',
                            'element_info': {'selector': element_selector}
                        }
                elif method == 'direct_url' and target_element and target_element.tag_name == 'img':
                    src = target_element.get_attribute('src')
                    if src:
                        result = self.capture_from_url(src)
                        if result is not None:
                            return {
                                'image': result,
                                'method': 'direct_url',
                                'element_info': {
                                    'selector': element_selector,
                                    'src': src
                                }
                            }
                else:
                    # Use smart capture as fallback
                    result = self.smart_capture(driver, target_element)
                    if result['image'] is not None:
                        return result
                
                # If all methods fail, try basic screenshot
                img = self.capture_visible_region(driver)
                if img is not None:
                    return {
                        'image': img,
                        'method': 'screenshot_fallback',
                        'element_info': {}
                    }
                
                return {'image': None, 'method': 'failed', 'element_info': {}}
                
            finally:
                driver.quit()
                
        except Exception as e:
            self.logger.error(f"Selenium capture failed: {e}")
            return {'image': None, 'method': 'error', 'error': str(e), 'element_info': {}}

    def capture_desktop_screenshot(self) -> Optional[np.ndarray]:
        """
        Capture desktop screenshot using PIL ImageGrab
        Useful for capturing dynamic content not accessible via Selenium
        """
        try:
            # Capture entire screen
            screenshot = ImageGrab.grab()
            # Convert PIL Image to OpenCV format
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            self.logger.error(f"Desktop screenshot failed: {e}")
            return None

    def resize_image(self, image: np.ndarray, max_width: int = 1920, max_height: int = 1080) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: Input image
            max_width: Maximum width
            max_height: Maximum height
            
        Returns:
            Resized image
        """
        if image is None:
            return None
            
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(max_width / width, max_height / height, 1.0)
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized
        
        return image

    def preprocess_image(self, image: np.ndarray, 
                        resize: bool = True,
                        denoise: bool = True,
                        enhance: bool = False) -> np.ndarray:
        """
        Preprocess captured image for better OCR results
        
        Args:
            image: Input image
            resize: Whether to resize large images
            denoise: Whether to apply denoising
            enhance: Whether to enhance contrast
            
        Returns:
            Preprocessed image
        """
        if image is None:
            return None
            
        processed = image.copy()
        
        # Resize if too large
        if resize:
            processed = self.resize_image(processed)
        
        # Convert to grayscale for some operations
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        if denoise:
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
        
        # Enhance contrast
        if enhance:
            # Convert to LAB color space
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge back and convert to BGR
            lab = cv2.merge([l, a, b])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return processed

    def extract_region(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Extract specific region from image using bounding box
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2] in normalized coordinates (0-1)
            
        Returns:
            Extracted region
        """
        if image is None:
            return None
            
        height, width = image.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # Extract region
        region = image[y1:y2, x1:x2]
        return region

    def save_image(self, image: np.ndarray, filepath: str) -> bool:
        """
        Save image to file
        
        Args:
            image: Image to save
            filepath: Output file path
            
        Returns:
            Success status
        """
        try:
            success = cv2.imwrite(filepath, image)
            return success
        except Exception as e:
            self.logger.error(f"Image save failed: {e}")
            return False

    def load_image(self, filepath: str) -> Optional[np.ndarray]:
        """
        Load image from file
        
        Args:
            filepath: Input file path
            
        Returns:
            Loaded image or None if failed
        """
        try:
            image = cv2.imread(filepath)
            return image
        except Exception as e:
            self.logger.error(f"Image load failed: {e}")
            return None

    def get_image_info(self, image: np.ndarray) -> Dict:
        """
        Get basic information about image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with image information
        """
        if image is None:
            return {}
            
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'size_kb': (image.nbytes if hasattr(image, 'nbytes') else 0) / 1024,
            'dtype': str(image.dtype)
        }

# Global instance
image_capture = ImageCapture()
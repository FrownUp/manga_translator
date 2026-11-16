# website_structure_analyzer.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import Dict, List, Any
import json
import time
from urllib.parse import urlparse

class WebsiteStructureAnalyzer:
    def __init__(self, headless: bool = True):
        self.options = Options()
        if headless:
            self.options.add_argument("--headless")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--window-size=1920,1080")
        self.driver = None
        
    def start_browser(self):
        """Initialize the headless browser"""
        self.driver = webdriver.Chrome(options=self.options)
        
    def close_browser(self):
        """Close the browser instance"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            
    def analyze_website(self, url: str, wait_time: int = 5) -> Dict[str, Any]:
        """
        Analyze the structure of a website and return element information
        that can be used for frontend overlays.
        """
        if not self.driver:
            self.start_browser()
            
        try:
            self.driver.get(url)
            # Wait for page to load
            WebDriverWait(self.driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Scroll through the page to trigger lazy loading
            self._scroll_page()
            
            # Get website structure
            structure = {
                "url": url,
                "domain": urlparse(url).netloc,
                "title": self.driver.title,
                "elements": self._get_dom_structure(),
                "viewport_size": self.driver.get_window_size(),
                "screenshot": self._take_screenshot()  # Base64 encoded screenshot
            }
            
            return structure
            
        except Exception as e:
            return {"error": str(e)}
    
    def _scroll_page(self):
        """Scroll through the entire page to trigger any lazy loading"""
        scroll_pause_time = 0.5
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        while True:
            # Scroll down to bottom
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # Wait to load page
            time.sleep(scroll_pause_time)
            
            # Calculate new scroll height and compare with last scroll height
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            
        # Scroll back to top
        self.driver.execute_script("window.scrollTo(0, 0);")
    
    def _get_dom_structure(self) -> List[Dict[str, Any]]:
        """Extract DOM structure with elements that are typically interactive"""
        elements = []
        
        # Define selectors for important elements
        selectors = {
            "buttons": "button, input[type='button'], input[type='submit']",
            "links": "a",
            "images": "img",
            "forms": "form, input, textarea, select",
            "containers": "div, section, article, header, footer, main, nav, aside",
            "text_elements": "h1, h2, h3, h4, h5, h6, p, span"
        }
        
        for element_type, selector in selectors.items():
            found_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
            
            for elem in found_elements:
                try:
                    # Get element properties
                    if elem.is_displayed():
                        location = elem.location
                        size = elem.size
                        elements.append({
                            "type": element_type,
                            "tag": elem.tag_name,
                            "id": elem.get_attribute("id") or "",
                            "classes": elem.get_attribute("class") or "",
                            "text": elem.text[:100] if elem.text else "",  # Limit text length
                            "position": {
                                "x": location['x'],
                                "y": location['y'],
                                "width": size['width'],
                                "height": size['height']
                            },
                            "visible": elem.is_displayed()
                        })
                except Exception:
                    # Skip elements that are no longer in DOM or not accessible
                    continue
        
        return elements
    
    def _take_screenshot(self) -> str:
        """Take a screenshot of the current page and return as base64"""
        return self.driver.get_screenshot_as_base64()
    
    def get_element_path(self, x: int, y: int) -> List[Dict[str, Any]]:
        """
        Get the hierarchy of elements at a specific coordinate.
        Useful for determining where to place overlays.
        """
        if not self.driver:
            return []
            
        # Use JavaScript to find elements at specific coordinates
        js_script = """
        var x = arguments[0];
        var y = arguments[1];
        var elements = [];
        var elem = document.elementFromPoint(x, y);
        
        while (elem) {
            var style = window.getComputedStyle(elem);
            elements.push({
                tag: elem.tagName,
                id: elem.id,
                classes: elem.className,
                visible: style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0'
            });
            elem = elem.parentElement;
        }
        return elements;
        """
        
        elements = self.driver.execute_script(js_script, x, y)
        return elements or []
    
    def generate_overlay_data(self, url: str) -> Dict[str, Any]:
        """
        Generate data structure that can be used by frontend to create overlays
        """
        structure = self.analyze_website(url)
        
        if "error" in structure:
            return structure
            
        # Filter for elements that are good candidates for overlays
        overlay_elements = []
        for elem in structure["elements"]:
            if (elem["visible"] and 
                elem["position"]["width"] > 10 and 
                elem["position"]["height"] > 10):
                overlay_elements.append(elem)
        
        return {
            "url": structure["url"],
            "title": structure["title"],
            "viewport_size": structure["viewport_size"],
            "screenshot": structure["screenshot"],
            "overlay_elements": overlay_elements,
            "timestamp": time.time()
        }


# Example usage and integration with your existing system
if __name__ == "__main__":
    # Example of how to integrate with your Flask server
    analyzer = WebsiteStructureAnalyzer()
    
    # Example analysis
    result = analyzer.generate_overlay_data("https://example.com")
    
    # Save result to a file or send to frontend
    with open("website_structure.json", "w") as f:
        json.dump(result, f, indent=2)
    
    analyzer.close_browser()
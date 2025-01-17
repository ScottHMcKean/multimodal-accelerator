from googlesearch import search
import requests
import os
import magic
import hashlib
from urllib.parse import urlparse
from tqdm import tqdm
import time

class SafeDocumentFinder:
    def __init__(self):
        self.download_dir = "downloaded_documents"
        self.supported_extensions = ('.pdf', '.docx', '.ppt', '.pptx')
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        os.makedirs(self.download_dir, exist_ok=True)
        
    def search_documents(self, keywords, max_results=100):
        """Search for documents using keywords"""
        search_query = f"{' '.join(keywords)} filetype:pdf OR filetype:docx OR filetype:ppt OR filetype:pptx"
        
        print(f"Searching for: {search_query}")
        urls = []
        
        try:
            # Use Google Search with delay to be polite
            for url in tqdm(search(search_query, num=max_results), desc="Finding documents"):
                if url.lower().endswith(self.supported_extensions):
                    urls.append(url)
                time.sleep(2)  # Be nice to Google
                
        except Exception as e:
            print(f"Search error: {e}")
            
        return urls

    def is_safe_file(self, content, url):
        """Basic safety checks for downloaded content"""
        # Check file size
        if len(content) > self.max_file_size:
            return False
            
        # Check if URL looks suspicious
        parsed_url = urlparse(url)
        suspicious = ['cgi-bin', 'php', 'script', 'download.php']
        if any(s in parsed_url.path.lower() for s in suspicious):
            return False
            
        return True

    def download_documents(self, urls):
        """Safely download documents from URLs"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        for url in tqdm(urls, desc="Downloading documents"):
            try:
                # Download with timeout
                response = requests.get(url, headers=headers, timeout=30, stream=True)
                if response.status_code != 200:
                    continue

                content = response.content
                
                # Safety checks
                if not self.is_safe_file(content, url):
                    print(f"Skipping potentially unsafe file: {url}")
                    continue

                # Generate safe filename using hash
                file_hash = hashlib.md5(content).hexdigest()
                extension = os.path.splitext(url)[1].lower()
                filename = f"{file_hash}{extension}"
                filepath = os.path.join(self.download_dir, filename)

                # Save file if it doesn't exist
                if not os.path.exists(filepath):
                    with open(filepath, 'wb') as f:
                        f.write(content)
                    print(f"Downloaded: {filename}")
                    
                time.sleep(1)  # Polite delay between downloads

            except Exception as e:
                print(f"Error downloading {url}: {e}")
import base64
from io import BytesIO

def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    buffered.seek(0)
    return base64.b64encode(buffered.read()).decode('utf-8')
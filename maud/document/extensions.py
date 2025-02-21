from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO


def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    buffered.seek(0)
    return base64.b64encode(buffered.read()).decode("utf-8")


def get_openai_description(
    client: OpenAI,
    model: str,
    image: Image.Image,
    image_type: str = "page",
    max_tokens: int = 200,
):
    assert image_type in ["page", "table", "picture"]

    img_bytes = encode_pil_image_to_base64(image)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Describe the contents of this {image_type}. Keep the response within {max_tokens} words",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_bytes}"},
                    },
                ],
            }
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

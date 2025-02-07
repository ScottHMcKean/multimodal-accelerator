from .utils import encode_pil_image_to_base64
from openai import OpenAI
from PIL import Image


def get_open_ai_image_description(
    client: OpenAI, image: Image.Image, image_type: str = "page", max_tokens: int = 200
):
    assert image_type in ["page", "table", "picture"]

    img_bytes = encode_pil_image_to_base64(image)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
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

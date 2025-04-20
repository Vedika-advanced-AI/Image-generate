from transformers import CLIPProcessor, CLIPModel
from PIL import Image


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def is_safe_image(
    model,
    processor,
    image,
):
    # Load image
    # image = Image.open(
    #     r"F:\om\2025\fastsdcpumcp\fastsdcpu\results\829a2123-92c8-4957-ad2f-06365a19665a-1.png"
    # )
    categories = ["safe", "nsfw"]
    inputs = processor(
        text=categories,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    safe_prob = dict(zip(categories, probs[0].tolist()))
    print(safe_prob)
    return safe_prob["safe"] > safe_prob["nsfw"]

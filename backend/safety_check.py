from transformers import pipeline


def is_safe_image(
    classifier,
    image,
):
    pred = classifier(image)
    nsfw_score = 0
    normal_score = 0
    for label in pred:
        if label["label"] == "nsfw":
            nsfw_score = label["score"]
        elif label["label"] == "normal":
            normal_score = label["score"]
    print(f"nsfw_score: {nsfw_score}, normal_score: {normal_score}")
    return normal_score > nsfw_score

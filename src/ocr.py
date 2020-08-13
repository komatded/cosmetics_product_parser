import base64
import pyheif
import requests
from PIL import Image


def resize(image, threshold):
    w, h = image.size
    if w > threshold or h > threshold:
        new_w = w * threshold / max(w, h)
        new_h = h * threshold / max(w, h)
        image = image.resize((int(new_w), int(new_h)), Image.ANTIALIAS)
    return image


def run_ocr(image_path: str, resize_threshold=1000):
    if image_path.endswith('.heic'):
        heif_file = pyheif.read_heif(image_path)
        image = Image.frombytes(mode=heif_file.mode, size=heif_file.size, data=heif_file.data)
        image = resize(image, threshold=resize_threshold)
    else:
        image = Image.open(image_path)
    image = resize(image, threshold=resize_threshold)
    image.save("../resources/temp.jpg", "JPEG")
    image_path = "../resources/temp.jpg"
    encoded_image = open(image_path, 'rb').read()
    encoded_image = base64.b64encode(encoded_image)
    return requests.post('http://188.246.224.225:8420/ocr', json={'image': encoded_image}).json()['answer']


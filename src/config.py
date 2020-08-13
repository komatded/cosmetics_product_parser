import os

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCE_DIR = os.path.abspath(os.path.join(MAIN_DIR, 'resources'))
IMAGE_DIR = os.path.abspath(os.path.join(RESOURCE_DIR, 'images'))
MODEL_PATH = os.path.abspath(os.path.join(RESOURCE_DIR, 'ProductClassifier.pickle'))

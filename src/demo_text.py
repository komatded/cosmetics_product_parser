from model import ProductClassifier
from ocr import run_ocr
from config import *
import argparse
import time

PC = ProductClassifier()
PC.load(MODEL_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images.')
    parser.add_argument('-i', '--image', type=str, help='images of products to identify')
    args = parser.parse_args()

    output = list()
    start = time.time()
    ocr_result = run_ocr(args.image)
    print('Ocr time: {0} sec.'.format(round(time.time() - start, 2)))
    product_name, class_probabilities = PC.predict(ocr_result)
    print('Full time: {0} sec.'.format(round(time.time() - start, 2)))
    print(product_name, class_probabilities.max())

import glob
import tqdm
import pandas as pd
from ocr import run_ocr_spin
from ast import literal_eval


def preprocess_text(text: str, stop_symbols=None):
    text = text.lower()
    if not stop_symbols:
        stop_symbols = {'\n', ',', '-'}
    return ''.join([i[0] for i in zip(text, text[1:] + '#') if i[0] != i[1] and i[0] not in stop_symbols])


def load_data(renew=False):
    if renew:
        data = list()
        products_images = glob.glob('../resources/train_data/*/*.*')
        for product_image in tqdm.tqdm(products_images):
            product_name = product_image.split('/')[-2].lower()
            ocr_result = run_ocr_spin(product_image)
            data.append({'product_name': product_name,
                         'ocr_result_90': ocr_result[0], 'ocr_result_180': ocr_result[1],
                         'ocr_result_270': ocr_result[2], 'ocr_result_0': ocr_result[3]})
        data = pd.DataFrame(data)
    else:
        columns = ['ocr_result_90', 'ocr_result_180', 'ocr_result_270', 'ocr_result_0']
        data = pd.read_csv('../resources/ocr_train_data.csv').fillna('[]')
        data[columns] = data[columns].applymap(lambda i: literal_eval(i))
    return data


def to_text_data(data):
    columns = ['ocr_result_90', 'ocr_result_180', 'ocr_result_270', 'ocr_result_0']
    data[columns] = data[columns].applymap(lambda i: [preprocess_text(ii[1]) for ii in i])
    data[columns] = data[columns].applymap(lambda i: [ii for ii in i if len(ii) > 2])
    data['words'] = data[columns].apply(lambda row: row[columns].sum(), axis=1)
    return data


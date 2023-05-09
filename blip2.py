from PIL import Image
import requests
import torch
from lavis.models import load_model_and_preprocess


def main():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(url, stream=True).raw)




if __name__ == '__main__':
    main()
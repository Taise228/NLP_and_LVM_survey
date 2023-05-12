from PIL import Image
import requests
import torch
from lavis.models import load_model_and_preprocess


def main():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(url, stream=True).raw)
    
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True)
    print('model downloaded')

    image = vis_processors['eval'](raw_image).unsqueeze(0)

    print(type(image))

    model.generate({'image': image, 'prompt': 'Question: what are in the picture? Answer:'})



if __name__ == '__main__':
    main()
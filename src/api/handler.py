import os 
import base64
import time
from PIL import Image

import torch
from fastapi import HTTPException
from pydantic import BaseModel
from torchvision.models import efficientnet_b0
from torchvision import datasets, transforms



class MineralImage(BaseModel):
    uploaded_image: bytes

class ModelHandler:

    def __init__(self):
        self.model = None
        self.is_loaded = False

    def load_model(self):

        some_path_todo = ""

        # TODO load model here and set self.model
        self.model = efficientnet_b0
        self.model.load_state_dict(torch.load(some_path_todo, weights_only=True))
        self.model.eval()

        self.is_loaded = True
        # TODO maybe add the matching values to the minerals here


    def predict_mineral(self, image_package: MineralImage):
        print('got here???')
        decoded_image = self.check_encoding(image_package.uploaded_image)
        print('oh it decoded')
        print(os.getcwd())
        temp = open('test_image/temp.jpeg', 'wb')
        temp.write(decoded_image)
        temp.close()

        # TODO implement transforms properly to make it work
        
        # TODO find why its breaking with a PIL.UnidentifiedImageError: cannot identify image file 'test_image/temp.jpeg'
        image = Image.open("test_image/temp.jpeg")
        transform = transforms.Compose([
            transforms.PILToTensor()
        ]) # with open "test_image"
        image_tensor = transform(image)

        image_tensor = image_tensor.to(torch.device("cpu"), non_blocking=True)
        result = self.model(image)
        # ???????????????? foul magic
        return {'result': result}

        # TODO call model to predict stuff post processing/checking image

    @staticmethod
    def check_encoding(base64_image: bytes):
        try:
            result = base64.b64decode(base64_image)
        except Exception as error:
            raise HTTPException(
                status_code = 400,
                detail = "incorrectly encoded image file",
                error = str(error), 
            )

        return result




model_obj = ModelHandler()
import sys
import json
import traceback

import uvicorn
import requests
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Form
from person_det_pose_edet4_trtserver import run_pdet_pose


app = FastAPI(title="Human body proportion length estimation")


class InputModel(BaseModel):
    """
    A Pydantic model that represents the input data for the body proportion length estimation API.

    Attributes:
        back_url (str): An optional URL to which the response will be sent via POST request.
        threshold (float): The detection threshold for the human detection model. Default is 0.80.
        person_height (int): The height of the person in centimeters. Default is 175 cm.
        image_file (bytes): The image file in bytes on which the estimation will be performed.
    """
    back_url: str = None
    threshold: float = 0.80
    person_height: int = 175
    image_file: bytes


class ModelProcessTask():
    """
    A class to handle the processing of the model inference asynchronously.

    Attributes:
        func (callable): The function to run for inference.
        input_data (InputModel): The input data required for the inference.
    """

    def __init__(self, func, input_data):
        super(ModelProcessTask, self).__init__()
        self.func = func
        self.input_data = input_data
        self.response_data = {}

    def run(self):
        """
        Executes the model inference with the below function & input data, & handles the response.
        """
        input_image_file = self.input_data.image_file
        threshold = self.input_data.threshold
        person_height = self.input_data.person_height

        result = self.func(
            input_image_file,
            det_threshold=threshold,
            person_height=[person_height],
            grpc_port="8081",  # GRPC is always set to this
            debug=False)
        self.response_data["code"] = "success"
        if len(result) == 0 or len(result[0]) < 3:
            self.response_data["msg"] = "No humans detected"
            body_proportion_lengths = {}
        else:
            self.response_data["msg"] = "human body proportion estimation complete"
            body_proportion_lengths = result[0][2]
        self.response_data["body_proportion_lengths_(cm)"] = body_proportion_lengths

        try:
            if self.input_data.back_url is not None:
                headers = {"Content-Type": "application/json"}
                print("RESPONSE DATA", self.response_data)
                requests.request(
                    method="POST",
                    url=self.input_data.back_url,
                    headers=headers,
                    data=json.dumps(self.response_data),
                    timeout=(3, 100))
                print("successfully sent")
        except Exception as e:
            traceback.print_exc()
            print(e)


@app.post("/body_proportion_length_estimation_file")
async def body_proportion_length_est_file(file: UploadFile = File(...),
                                          person_height_in_cm: int = Form(175),
                                          threshold: float = Form(0.70)):
    """
    Endpoint for estimating body proportion lengths from an uploaded file.

    Parameters:
        file (UploadFile): The image file uploaded by the user.
        person_height_in_cm (int): The height of the person in centimeters.
        threshold (float): The detection threshold.

    Returns:
        dict: The response data containing the estimation result or an error message.
    """
    response_data = {}
    try:
        # send iamge directly to demo func as bytes
        file_bytes_content = file.file.read()
        input_data = InputModel(
            person_height=person_height_in_cm,
            back_url=None,
            threshold=threshold,
            image_file=file_bytes_content)
        task = ModelProcessTask(run_pdet_pose,
                                input_data=input_data)
        task.run()
        response_data = task.response_data

    except Exception as e:
        traceback.print_exc()
        print(e)
        response_data["msg"] = "Failed to run inference on image. Please use an image with one fully visible human."
        response_data["code"] = "failed"
    return response_data


@app.get("/")
def index():
    return {"Welcome to Human Body Proportion Estimation Web Service": "Please visit /docs"}


if __name__ == '__main__':
    if len(sys.argv) == 1:
        uvicorn.run("server:app", host='0.0.0.0',
                    port=8080, workers=1)

    elif len(sys.argv) == 2:
        print("Using port: " + sys.argv[1])
        uvicorn.run("server:app", host='0.0.0.0',
                    port=int(sys.argv[1]), workers=1)

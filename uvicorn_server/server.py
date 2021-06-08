from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import traceback
import requests
import uvicorn
import json
import sys
from person_det_pose_est_trtserver_demo import run_pdet_pose

app = FastAPI(title="Human body proportion length estimation")


class InputModel(BaseModel):
    back_url: str = None
    threshold: float = 0.80
    person_height: int = 175
    image_file: bytes


class ModelProcessTask():
    def __init__(self, func, input_data):
        super(ModelProcessTask, self).__init__()
        self.func = func
        self.input_data = input_data
        self.response_data = dict()

    def run(self):
        input_image_file = self.input_data.image_file
        threshold = self.input_data.threshold
        person_height = self.input_data.person_height

        self.result = self.func(input_image_file,
                                det_threshold=threshold,
                                person_height=[person_height],
                                grpc_port="8081",  # grpc is always set to this
                                debug=False)
        self.response_data["code"] = "success"
        if len(self.result) == 0 or len(self.result[0]) < 3:
            self.response_data["msg"] = "No humans detected"
            body_proportion_lengths = {}
        else:
            self.response_data["msg"] = "human body proportion estimation complete"
            body_proportion_lengths = self.result[0][2]
        self.response_data["body_proportion_lengths_(cm)"] = body_proportion_lengths

        try:
            if self.input_data.back_url is not None:
                headers = {"Content-Type": "application/json"}
                print("RESPONSE DATA", self.response_data)
                requests.request(method="POST",
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
    response_data = dict()
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

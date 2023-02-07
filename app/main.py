from fastapi import FastAPI, UploadFile, File
from PIL import Image
from app.loading import load_model, get_device, get_transforms
import torch.nn.functional as F
import json
from fastapi.middleware.cors import CORSMiddleware


model = load_model()
device = get_device()
transform = get_transforms()

print(device)

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/image")
async def recognize_emotions(img: UploadFile = File(...)):
    original_image = Image.open(img.file)
    input = transform(original_image)
    input = input.unsqueeze_(0)
    model.eval()
    output = model(input.to(device))
    output = F.softmax(output, dim = 1)
    result = output.cpu().detach().numpy()[0]
    return json.dumps(dict(enumerate(map(str, result.tolist()))))


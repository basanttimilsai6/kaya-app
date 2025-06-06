from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil

from app.config import settings
from app.services import analyze_skin_to_json

app = FastAPI(title=settings.app_name, version=settings.app_version)

MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)

@app.post("/upload-image/")
def upload_image(file: UploadFile = File(...)):
    file_location = MEDIA_DIR / file.filename

    with file_location.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_skin_to_json(file_location)

    return JSONResponse(content=result, status_code=200)

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any
import traceback


from text_processing import process_uploaded_file_logic, TextProcessingError 

app = FastAPI()
templates = Jinja2Templates(directory="templates")


MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
CHUNK_SIZE_BYTES = 1 * 1024 * 1024
N_ROWS_OUTPUT = 50


@app.get("/")
async def main_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/")
async def process_file_endpoint(request: Request, file: UploadFile = File(...)):
    results_for_template: List[Dict[str, Any]] = []
    error_message: str | None = None
    filename: str | None = file.filename

    try:
        results_for_template = await process_uploaded_file_logic(
            file,
            MAX_FILE_SIZE_BYTES,
            CHUNK_SIZE_BYTES,
            N_ROWS_OUTPUT
        )
    except TextProcessingError as tpe:
        error_message = str(tpe)
    except Exception as e:
        traceback.print_exc()
        error_message = "Произошла внутренняя ошибка сервера при обработке вашего запроса."
        results_for_template = []


    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": results_for_template,
            "filename": filename,
            "error_message": error_message,
        },
    )
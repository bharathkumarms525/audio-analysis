from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from app1.services.transcription import transcribe_audio_file

router = APIRouter()

@router.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    response = await transcribe_audio_file(file)
    return JSONResponse(content=response)



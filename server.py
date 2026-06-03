import os
import shutil
import tempfile
from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
import uvicorn

from graph import build_graph
from state import FeedGenerationState

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="도자기 인스타그램 피드 생성기 API")

# 생성된 이미지 파일 서빙
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/generate")
async def generate_feed(
    image: UploadFile = File(...),
    style: str = Form("warm"),
    user_text: str = Form(""),
):
    suffix = os.path.splitext(image.filename or "upload.jpg")[1] or ".jpg"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            shutil.copyfileobj(image.file, f)

        initial_state: FeedGenerationState = {
            "original_image_path": tmp_path,
            "selected_style": style,
            "user_text": user_text.strip() if user_text else None,
            "image_analysis": {},
            "mood": "",
            "generation_prompt": "",
            "generated_image_url": "",
            "caption": "",
            "hashtags": [],
            "current_status": "initialized",
            "retry_count": 0,
            "error_message": None,
        }

        pipeline = build_graph()
        final_state = await run_in_threadpool(pipeline.invoke, initial_state)

        generated_path = final_state.get("generated_image_url", "")
        image_url = ""
        if generated_path and os.path.exists(generated_path):
            filename = os.path.basename(generated_path)
            image_url = f"/output/{filename}"

        return JSONResponse({
            "caption": final_state.get("caption", ""),
            "hashtags": final_state.get("hashtags", []),
            "image_url": image_url,
            "mood": final_state.get("mood", ""),
            "error": final_state.get("error_message"),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

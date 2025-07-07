import io
import os
from typing import Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import easyocr


def detect_circles(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    raw = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=80,
        param1=100, param2=50,
        minRadius=20, maxRadius=120
    )
    if raw is None:
        return np.zeros((0,3), dtype=int)
    
    circles = np.round(raw[0]).astype(int)
    
    unique = []
    for x, y, r in circles:
        if not any((x - x2)**2 + (y - y2)**2 < (min(r, r2) * 0.5)**2 for x2, y2, r2 in unique):
            unique.append((x, y, r))
    return np.array(unique, dtype=int)

app = FastAPI(title="Piston Counter API (EasyOCR)")


reader = easyocr.Reader(['en'], gpu=False)

@app.post("/count_pistons/", response_model=Dict[str,int])
async def count_pistons(file: UploadFile = File(...)):
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        raise HTTPException(status_code=415, detail="Only JPG/PNG images supported")

    
    data = await file.read()
    
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    circles = detect_circles(bgr)
    
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    counts = {f"P{i}": 0 for i in range(1,5)}
    
    for x, y, r in circles:
        m = int(r * 0.2)
        x0, x1 = max(0, x-r-m), min(gray.shape[1], x+r+m)
        y0, y1 = max(0, y-r-m), min(gray.shape[0], y+r+m)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        
        pil_roi = Image.fromarray(roi)
        results = reader.readtext(np.array(pil_roi), detail=1, allowlist="P1234")
        if results:
            
            _, raw_txt, best_conf = max(results, key=lambda item: item[2])
            txt = raw_txt.strip().upper()
            
            if txt in counts and best_conf >= 0.50:
                counts[txt] += 1
    return counts

@app.post("/count_pistons/debug")
async def debug_pistons(file: UploadFile = File(...)):
    data = await file.read()
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    circles = detect_circles(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    counts = {f"P{i}": 0 for i in range(1,5)}
    debug = []
    for x, y, r in circles:
        m = int(r * 0.2)
        x0, x1 = max(0, x-r-m), min(gray.shape[1], x+r+m)
        y0, y1 = max(0, y-r-m), min(gray.shape[0], y+r+m)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        
        results = reader.readtext(np.array(roi), detail=1, allowlist="P1234")
        label = None
        t = None
        conf = None
        if results:
            
            _, raw_txt, best_conf = max(results, key=lambda item: item[2])
            t = raw_txt.strip().upper()
            if t in counts and best_conf >= 0.50:
                counts[t] += 1
                label = t
            conf = best_conf
        debug.append({"x": int(x), "y": int(y), "r": int(r), "label": label, "text": t, "conf": float(conf) if conf is not None else None})
    return JSONResponse(content={"detections": debug, "counts": counts})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
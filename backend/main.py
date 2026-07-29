"""FQC Sleep + Eye Direction Detection API."""

from __future__ import annotations

import os
import time

from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import Event, SessionLocal, get_db
from engine import VIDEOS_DIR, processor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "data", "snapshots")

app = FastAPI(title="FQC Sleep & Eye Direction Detection", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/snapshots", StaticFiles(directory=SNAPSHOTS_DIR), name="snapshots")


class OverlayUpdate(BaseModel):
    landmarks: bool | None = None
    gaze_arrow: bool | None = None
    hud: bool | None = None


class SourceUpdate(BaseModel):
    source: str  # filename under sample_videos, or "webcam"


class StatusUpdate(BaseModel):
    status: str


@app.on_event("startup")
def on_startup():
    videos = processor.list_videos()
    if videos:
        processor.set_source(videos[0])
    else:
        processor.set_source("webcam")
    processor.start()


@app.on_event("shutdown")
def on_shutdown():
    processor.stop()


def generate_frames():
    processor.register_stream_client()
    last = 0
    try:
        while True:
            jpeg, last = processor.get_jpeg(last, timeout=1.0)
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
    finally:
        processor.unregister_stream_client()


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stats")
def api_stats(db: Session = Depends(get_db)):
    live = processor.get_stats()
    rows = db.query(Event.event_type, func.count(Event.id)).group_by(Event.event_type).all()
    counts = {k: v for k, v in rows}
    return {
        **live,
        "total_alerts": sum(counts.values()),
        "sleep_alerts": counts.get("Sleep", 0),
        "looking_away_alerts": counts.get("LookingAway", 0),
        "tired_alerts": counts.get("Tired", 0),
    }


@app.get("/api/events")
def api_events(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    rows = (
        db.query(Event)
        .order_by(Event.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": e.id,
            "event_type": e.event_type,
            "severity": e.severity,
            "message": e.message,
            "snapshot_path": e.snapshot_path,
            "status": e.status,
            "created_at": e.created_at.isoformat() + "Z" if e.created_at else None,
        }
        for e in rows
    ]


@app.patch("/api/events/{event_id}")
def patch_event(event_id: int, body: StatusUpdate, db: Session = Depends(get_db)):
    evt = db.query(Event).filter(Event.id == event_id).first()
    if not evt:
        return {"ok": False, "error": "not found"}
    evt.status = body.status
    db.commit()
    return {"ok": True}


@app.delete("/api/events")
def clear_events(db: Session = Depends(get_db)):
    db.query(Event).delete()
    db.commit()
    return {"ok": True}


@app.get("/api/videos")
def api_videos():
    return {"videos": processor.list_videos(), "current": str(processor.source)}


@app.post("/api/source")
def api_source(body: SourceUpdate):
    processor.stop()
    processor.set_source(body.source)
    processor.start()
    return {"ok": True, "source": str(processor.source)}


@app.post("/api/overlays")
def api_overlays(body: OverlayUpdate):
    processor.set_overlays(
        **{k: v for k, v in body.model_dump().items() if v is not None}
    )
    return {"ok": True, "overlays": processor.get_stats()["overlays"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8010, reload=False)

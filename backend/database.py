"""SQLite persistence for FQC attention alerts."""

from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "events.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, index=True)  # Sleep | LookingAway | Tired
    severity = Column(String, default="High")
    message = Column(String)
    snapshot_path = Column(String, nullable=True)
    status = Column(String, default="New")
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_event(
    db: Session,
    event_type: str,
    message: str,
    severity: str = "High",
    snapshot_path: str | None = None,
) -> Event:
    evt = Event(
        event_type=event_type,
        severity=severity,
        message=message,
        snapshot_path=snapshot_path,
        status="New",
    )
    db.add(evt)
    db.commit()
    db.refresh(evt)
    return evt

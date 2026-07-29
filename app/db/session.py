"""SQLAlchemy engine and session infrastructure."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

engine = create_engine(get_settings().database_url, pool_pre_ping=True)
SessionLocal = sessionmaker[Session](bind=engine, expire_on_commit=False)

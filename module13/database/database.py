from pathlib import Path
from urllib.parse import quote_plus
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


# ============================================================
# DATABASE SETTINGS
# ============================================================

DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ccaa_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")


# ============================================================
# DATABASE URL
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Check your .env file.")


# ============================================================
# SQLALCHEMY ENGINE
# ============================================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=False
)


# ============================================================
# SESSION
# ============================================================

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# ============================================================
# BASE MODEL
# ============================================================

Base = declarative_base()


# ============================================================
# FASTAPI DATABASE DEPENDENCY
# ============================================================

def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


# ============================================================
# CONNECTION TEST
# ============================================================

def test_database_connection():
    try:
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT current_database();")
            )

            database_name = result.scalar()

            print("✅ PostgreSQL connection successful!")
            print(f"✅ Database: {database_name}")
            print(f"✅ Host: {DB_HOST}")
            print(f"✅ Port: {DB_PORT}")

            return True

    except Exception as error:
        print("❌ PostgreSQL connection failed.")
        print(f"Error: {error}")

        return False


# ============================================================
# RUN TEST
# ============================================================

if __name__ == "__main__":
    test_database_connection()
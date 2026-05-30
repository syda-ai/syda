"""Create a small SQLite demo database for the CLI example."""

import sqlite3
import sys

db_path = sys.argv[1] if len(sys.argv) > 1 else "demo.db"

conn = sqlite3.connect(db_path)
conn.executescript("""
    DROP TABLE IF EXISTS appointment;
    DROP TABLE IF EXISTS patient;
    DROP TABLE IF EXISTS provider;

    CREATE TABLE provider (
        id        INTEGER PRIMARY KEY,
        name      TEXT NOT NULL,
        specialty TEXT,
        license   TEXT
    );

    CREATE TABLE patient (
        id          INTEGER PRIMARY KEY,
        first_name  TEXT NOT NULL,
        last_name   TEXT NOT NULL,
        dob         DATE,
        provider_id INTEGER REFERENCES provider(id)
    );

    CREATE TABLE appointment (
        id          INTEGER PRIMARY KEY,
        patient_id  INTEGER NOT NULL REFERENCES patient(id),
        provider_id INTEGER NOT NULL REFERENCES provider(id),
        visit_date  DATE,
        status      TEXT
    );
""")
conn.commit()
conn.close()
print(f"Created demo DB: {db_path}")

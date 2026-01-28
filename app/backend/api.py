from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, PUT, etc.
    allow_headers=["*"],
)

@app.get("/api/test_instances")
def get_test_instances():
    response = {"instances": [
        {
            "name": "Seatwork-1",
            "section": "3-Rizal",
            "date": "2025-11-11T20:17:46.384Z",
            "test_id": "Seatwork-1_3-Rizal",
            "is_done_rendering": True
        },
        {
            "name": "Seatwork-2",
            "section": "3-Aguinaldo",
            "date": "2025-12-12T20:17:46.384Z",
            "test_id": "Seatwork-2_3-Aguinaldo",
            "is_done_rendering": False
        },
        {
            "name": "Quiz-1",
            "section": "3-Aguinaldo",
            "date": "2026-01-12T20:17:46.384Z",
            "test_id": "Quiz-1_3-Aguinaldo",
            "is_done_rendering": False
        },
        {
            "name": "Quiz-1",
            "section": "3-Rizal",
            "date": "2026-01-13T20:17:46.384Z",
            "test_id": "Quiz-1_3-Rizal",
            "is_done_rendering": True
        },
        {
            "name": "Quiz-2",
            "section": "3-Aguinaldo",
            "date": "2026-01-19T20:17:46.384Z",
            "test_id": "Quiz-2_3-Aguinaldo",
            "is_done_rendering": True
        }
    ]}
    return response
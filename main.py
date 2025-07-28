from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
from geopy.distance import geodesic

app = FastAPI()

# Allow requests from your mobile app (Expo or anything)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Airtable config
BASE_ID = "appApTB0N861wvwFU"
API_TOKEN = "patcu23BTWvHdAIM2.d4dd2ccce040582af9172c9622b6e6e07ed99f6b6c9ca4955a3eb7c0ef879d5b"  # Replace with your real token
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
TABLE_USER = "User Profiles"
TABLE_EXPERIENCES = "Experiences"

# Helper to pull from Airtable
def fetch_airtable_records(table_name):
    url = f"https://api.airtable.com/v0/{BASE_ID}/{table_name}"
    all_records = []
    offset = None

    while True:
        params = {"offset": offset} if offset else {}
        res = requests.get(url, headers=HEADERS, params=params)
        data = res.json()
        all_records.extend(data.get("records", []))
        offset = data.get("offset")
        if not offset:
            break

    return all_records

# Main endpoint
@app.post("/recommendations")
async def get_recommendations(request: Request):
    print("🚨 Incoming request:", await request.body())
    body = await request.json()

    # Required fields
    lat = body["lat"]
    lon = body["lon"]
    weather = body["weather"]
    time_available = body.get("time_available_mins", 60)
    transport_mode = body.get("transport_mode", "walking")

    max_distance = {
        "walking": 1.0,
        "transit": 3.0,
        "driving": 10.0
    }.get(transport_mode, 1.0)

    # Check for preferences passed in request
    user_interests = set(body.get("interests", []))
    user_vibes = set(body.get("vibes", []))

    # Fallback to Airtable user if preferences missing
    if not user_interests or not user_vibes:
        user_id = body.get("user_id", "user_001")
        users = fetch_airtable_records(TABLE_USER)
        for u in users:
            if u["fields"].get("user_id") == user_id:
                fields = u["fields"]
                user_interests = user_interests or set(fields.get("interests", []))
                user_vibes = user_vibes or set(fields.get("vibes", []))
                break

    experiences = fetch_airtable_records(TABLE_EXPERIENCES)
    user_location = (lat, lon)
    scored = []

    for rec in experiences:
        f = rec.get("fields", {})

        # Location parsing
        try:
            loc = tuple(map(float, f["lat_lng_clean"].split(",")))
        except:
            continue

        if not f.get("open_now", True):
            continue

        if f.get("time_estimate_mins", 0) > time_available:
            continue

        if f.get("weather_sensitive") == "avoid_rain" and weather in ["rain", "fog"]:
            continue

        distance = geodesic(user_location, loc).miles
        if distance > max_distance:
            continue

        interest_score = len(user_interests & set(f.get("interest_tags", [])))
        vibe_score = len(user_vibes & set(f.get("vibe_tags", [])))
        skip_penalty = 0.5 * f.get("skipped_count", 0)

        total_score = interest_score + vibe_score - skip_penalty

        scored.append({
            "title": f.get("title", "Unnamed"),
            "score": total_score,
            "distance_miles": round(distance, 2),
            "weather_sensitive": f.get("weather_sensitive"),
            "open_now": f.get("open_now")
        })

    return sorted(scored, key=lambda x: x["score"], reverse=True)

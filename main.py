from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import requests
from geopy.distance import geodesic
import math

# ------------------ FastAPI app ------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Airtable config ------------------
BASE_ID = "appApTB0N861wvwFU"
API_TOKEN = "patcu23BTWvHdAIM2.d4dd2ccce040582af9172c9622b6e6e07ed99f6b6c9ca4955a3eb7c0ef879d5b"  # replace or use env var in Render
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
TABLE_USER = "User Profiles"
TABLE_EXPERIENCES = "Experiences"

# ------------------ Request model ------------------
class RecRequest(BaseModel):
    lat: float
    lon: float
    weather: str = Field(..., description="lowercase e.g. 'clear','clouds','rain','fog','drizzle'")
    time_available_mins: int = 90
    transport_mode: str = "walking"  # walking | transit | driving
    time_of_day: Optional[str] = Field(None, description="morning|midday|afternoon|evening")
    # preferences (optional -> falls back to Airtable user if missing)
    user_id: Optional[str] = "user_001"
    interests: Optional[List[str]] = None
    vibes: Optional[List[str]] = None
    # accessibility / parent-perk preferences (optional)
    need_stroller_friendly: Optional[bool] = None
    want_food_nearby: Optional[bool] = None
    want_quiet_space: Optional[bool] = None
    want_less_crowded: Optional[bool] = None
    need_changing_station: Optional[bool] = None
    child_age_years: Optional[float] = None  # e.g., 0.5, 2, 4.5

# ------------------ Helpers ------------------
def fetch_airtable_records(table_name: str):
    url = f"https://api.airtable.com/v0/{BASE_ID}/{table_name}"
    all_records = []
    offset = None
    while True:
        params = {"offset": offset} if offset else {}
        res = requests.get(url, headers=HEADERS, params=params, timeout=20)
        res.raise_for_status()
        data = res.json()
        all_records.extend(data.get("records", []))
        offset = data.get("offset")
        if not offset:
            break
    return all_records

def parse_bool(value) -> Optional[bool]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"true","yes","y","1"}:
        return True
    if s in {"false","no","n","0"}:
        return False
    return None  # unknown stays None

def within_time(estimate: Optional[int], available: int) -> bool:
    if not estimate:
        return True
    return estimate <= available

def max_distance_for_transport(mode: str) -> float:
    return {
        "walking": 1.2,   # mi
        "transit": 3.5,
        "driving": 10.0,
    }.get(mode, 1.2)

def weather_bad_for_outdoor(weather: str) -> bool:
    return weather in {"rain","drizzle","thunderstorm","fog"}

def parse_latlng(s: str) -> Optional[Tuple[float,float]]:
    try:
        lat, lng = [float(x.strip()) for x in s.split(",")]
        return (lat, lng)
    except Exception:
        return None

def age_fits(best_age_range: Optional[str], child_age: Optional[float]) -> Optional[bool]:
    """
    best_age_range examples: "0-2", "2-5", "3+", "All"
    Returns True/False/None(unknown)
    """
    if child_age is None or not best_age_range:
        return None
    s = str(best_age_range).strip().lower()
    if s in {"all", "any"}:
        return True
    try:
        if "+" in s:
            lo = float(s.replace("+","").strip())
            return child_age >= lo
        if "-" in s:
            lo, hi = [float(x.strip()) for x in s.split("-")]
            return (child_age >= lo) and (child_age <= hi)
    except Exception:
        return None
    return None

def tod_fits(tod: Optional[str], hours_hint: Optional[str]) -> Optional[bool]:
    """
    hours_hint could be a loose text like 'best in morning' or blank; optional.
    If none, return None. If matches time_of_day, True; else False.
    """
    if not tod or not hours_hint:
        return None
    t = tod.strip().lower()
    h = str(hours_hint).strip().lower()
    if "morning" in h and t == "morning": return True
    if "midday" in h and t == "midday": return True
    if "afternoon" in h and t == "afternoon": return True
    if "evening" in h and t == "evening": return True
    # If hint exists but not matching requested tod, mark False to lightly down-rank
    return False

# ------------------ Endpoint ------------------
@app.post("/recommendations")
async def recommendations(req: RecRequest):
    # Load user prefs if missing
    interests = set([x.lower() for x in (req.interests or [])])
    vibes = set([x.lower() for x in (req.vibes or [])])
    if not interests or not vibes:
        users = fetch_airtable_records(TABLE_USER)
        for u in users:
            f = u.get("fields", {})
            if f.get("user_id") == req.user_id:
                if not interests:
                    interests = set([x.lower() for x in f.get("interests", [])])
                if not vibes:
                    vibes = set([x.lower() for x in f.get("vibes", [])])
                break

    experiences = fetch_airtable_records(TABLE_EXPERIENCES)

    # Precompute limits / knobs
    max_dist_mi = max_distance_for_transport(req.transport_mode)
    user_loc = (req.lat, req.lon)

    # Weights (tuneable)
    W_INTEREST = 3.0
    W_VIBE = 1.5
    W_DISTANCE = 2.5        # applied as smooth decay
    W_TIME_FIT = 1.0
    W_WEATHER = 1.5
    W_AGE = 2.0
    W_TOD = 0.75
    # Parent-joy perks
    W_STROLLER = 1.2
    W_FOOD = 1.3
    W_QUIET = 1.1
    W_LESS_CROWD = 1.1
    W_CHANGING = 1.4
    # Gentle penalty if previously skipped a lot
    W_SKIP = 0.4

    scored = []
    for rec in experiences:
        f = rec.get("fields", {})
        title = f.get("title", "Untitled")

        # --- Parse and basic filters ---
        latlng = parse_latlng(f.get("lat_lng_clean") or f.get("lat_lng") or "")
        if not latlng:
            continue

        # Open now (if provided)
        if f.get("open_now") is False:
            continue

        # Time fit
        estimate = f.get("time_estimate_mins") or f.get("estimated_time_mins") or 0
        if not within_time(estimate, req.time_available_mins):
            continue

        # Weather sensitivity
        weather_sensitive = (f.get("weather_sensitive") or "").strip().lower()  # "avoid_rain", "ok_any", etc.
        if weather_sensitive in {"avoid_rain","outdoor_only"} and weather_bad_for_outdoor(req.weather):
            # hard filter out truly bad combos
            continue

        # Distance gate
        dist_mi = geodesic(user_loc, latlng).miles
        if dist_mi > max_dist_mi:
            continue

        # --- Scoring ---
        score = 0.0
        details = {}

        # Interest & vibe overlap
        interest_tags = set([x.lower() for x in f.get("interest_tags", [])])
        vibe_tags = set([x.lower() for x in f.get("vibe_tags", [])])
        interest_overlap = len(interests & interest_tags)
        vibe_overlap = len(vibes & vibe_tags)
        score += W_INTEREST * interest_overlap
        score += W_VIBE * vibe_overlap
        details["interest_overlap"] = interest_overlap
        details["vibe_overlap"] = vibe_overlap

        # Smooth distance decay (closer is better)
        # 1.0 near, falls off towards max_dist
        if max_dist_mi > 0:
            proximity = max(0.0, 1.0 - (dist_mi / max_dist_mi))
            score += W_DISTANCE * proximity
            details["proximity_bonus"] = round(W_DISTANCE * proximity, 2)

        # Time-of-day hint
        tod_hint = f.get("best_time_of_day")  # optional text field you can populate
        tod_fit = tod_fits(req.time_of_day, tod_hint)
        if tod_fit is True:
            score += W_TOD
        elif tod_fit is False:
            score -= W_TOD * 0.5
        details["tod_fit"] = tod_fit

        # Weather gentle boost for indoor when bad weather
        is_indoorish = parse_bool(f.get("has_quiet_space")) or False  # proxy for calmer/indoor-ish
        if weather_bad_for_outdoor(req.weather) and is_indoorish:
            score += W_WEATHER
            details["weather_boost"] = "indoor_preferred"

        # Age fit
        fits_age = age_fits(f.get("best_age_range"), req.child_age_years)
        if fits_age is True:
            score += W_AGE
        elif fits_age is False:
            score -= W_AGE * 0.75
        details["age_fit"] = fits_age

        # Parent‑joy perks (based on user asks)
        stroller_friendly = parse_bool(f.get("stroller_friendly"))
        food_nearby = parse_bool(f.get("food_nearby"))
        quiet_space = parse_bool(f.get("has_quiet_space"))
        less_crowded = parse_bool(f.get("less_crowded_place"))
        changing_station = parse_bool(f.get("has_changing_station"))

        if req.need_stroller_friendly and stroller_friendly is True:
            score += W_STROLLER
        if req.want_food_nearby and food_nearby is True:
            score += W_FOOD
        if req.want_quiet_space and quiet_space is True:
            score += W_QUIET
        if req.want_less_crowded and less_crowded is True:
            score += W_LESS_CROWD
        if req.need_changing_station and changing_station is True:
            score += W_CHANGING

        # Skips/penalties
        skipped = f.get("skipped_count") or 0
        score -= W_SKIP * skipped

        scored.append({
            "title": title,
            "score": round(score, 2),
            "distance_miles": round(dist_mi, 2),
            "time_estimate_mins": estimate or None,
            "weather_sensitive": weather_sensitive or None,
            "best_age_range": f.get("best_age_range"),
            "interest_tags": list(interest_tags) if interest_tags else [],
            "vibe_tags": list(vibe_tags) if vibe_tags else [],
            "stroller_friendly": stroller_friendly,
            "food_nearby": food_nearby,
            "has_quiet_space": quiet_space,
            "less_crowded_place": less_crowded,
            "has_changing_station": changing_station,
            "debug": details,  # helpful while tuning
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


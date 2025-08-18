from fastapi import FastAPI, Request, HTTPException
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import os
import re, math, requests
from geopy.distance import geodesic
from fastapi.responses import JSONResponse
import traceback

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True, "hint": "Try /ping and /docs"}

@app.get("/ping")
def ping():
    return {"ok": True}

@app.post("/echo")
async def echo(request: Request):
    body = await request.json()
    return {"received": body}

@app.on_event("startup")
async def list_routes():
    print("Loaded routes:")
    for r in app.routes:
        try:
            print("  ", r.methods, r.path)
        except Exception:
            print("  ", r)

@app.exception_handler(Exception)
async def catch_all_exceptions(request: Request, exc: Exception):
    print("UNCAUGHT EXCEPTION:", repr(exc))
    traceback.print_exc()
    return JSONResponse(status_code=400, content={"detail": "Unhandled error", "type": str(type(exc).__name__)})


# ---------- Helpers (paste near your imports) ----------

LATLNG_RE = re.compile(r'^\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*$')
SF_CENTER = (37.7793, -122.4193)
BAY_RADIUS_MI = 60.0  # consider "Bay Area" within ~60mi of SF center

def parse_latlng(s):
    """Parse 'lat,lng' into (lat, lng) or return None."""
    if not isinstance(s, str):
        return None
    m = LATLNG_RE.match(s)
    if not m:
        return None
    lat = float(m.group(1)); lng = float(m.group(2))
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return None
    return (lat, lng)

def safe_distance_miles(a, b):
    """Return distance in miles or None if anything fails."""
    try:
        return geodesic(a, b).miles
    except Exception:
        return None

def is_within_bay_area(user_ll):
    """Check if coords are within a loose Bay Area radius of SF center."""
    d = safe_distance_miles(user_ll, SF_CENTER)
    return (d is not None) and (d <= BAY_RADIUS_MI)

# ------------------ Airtable config ------------------
BASE_ID = "appApTB0N861wvwFU"
API_TOKEN = os.environ.get("AIRTABLE_API_TOKEN", "")
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
    try:
        # ---------- Load user prefs if missing ----------
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

        # ---------- Precompute limits / knobs ----------
        max_dist_mi = max_distance_for_transport(req.transport_mode)

        # Validate and build user location
        if req.lat is None or req.lon is None:
            raise HTTPException(status_code=400, detail="Missing lat/lon in request.")
        try:
            user_loc = (float(req.lat), float(req.lon))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid lat/lon types; must be numbers.")

        # If you're testing from CT (outside Bay Area), fallback to SF center
        if not is_within_bay_area(user_loc):
            user_loc = SF_CENTER  # testing fallback

        # ---------- Weights (tuneable) ----------
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
            latlng_str = f.get("lat_lng_clean") or f.get("lat_lng") or ""
            latlng = parse_latlng(latlng_str)
            if not latlng:
                continue  # skip silently; bad coordinates should never 500

            # Open now (if provided)
            if f.get("open_now") is False:
                continue

            # Time fit
            estimate = f.get("time_estimate_mins") or f.get("estimated_time_mins") or 0
            if not within_time(estimate, req.time_available_mins):
                continue

            # Weather sensitivity
            weather_sensitive = (f.get("weather_sensitive") or "").strip().lower()  # "avoid_rain", "ok_any", etc.
            try:
                bad_outdoor = weather_bad_for_outdoor(req.weather)
            except Exception:
                bad_outdoor = False  # fail-open to avoid crashing on weather input shape

            if weather_sensitive in {"avoid_rain", "outdoor_only"} and bad_outdoor:
                continue  # hard filter out truly bad combos

            # Distance gate (guarded)
            dist_mi = safe_distance_miles(user_loc, latlng)
            if dist_mi is None:
                continue  # skip rows that can't compute distance
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
            if max_dist_mi > 0:
                proximity = max(0.0, 1.0 - (dist_mi / max_dist_mi))
                score += W_DISTANCE * proximity
                details["proximity_bonus"] = round(W_DISTANCE * proximity, 2)

            # Time-of-day hint
            tod_hint = f.get("best_time_of_day")  # optional text field
            try:
                tod_fit = tod_fits(req.time_of_day, tod_hint)
            except Exception:
                tod_fit = None
            if tod_fit is True:
                score += W_TOD
            elif tod_fit is False:
                score -= W_TOD * 0.5
            details["tod_fit"] = tod_fit

            # Weather gentle boost for indoor when bad weather
            is_indoorish = bool(parse_bool(f.get("has_quiet_space")))  # proxy for calmer/indoor-ish
            if bad_outdoor and is_indoorish:
                score += W_WEATHER
                details["weather_boost"] = "indoor_preferred"

            # Age fit
            try:
                fits_age = age_fits(f.get("best_age_range"), req.child_age_years)
            except Exception:
                fits_age = None
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

            if getattr(req, "need_stroller_friendly", False) and stroller_friendly is True:
                score += W_STROLLER
            if getattr(req, "want_food_nearby", False) and food_nearby is True:
                score += W_FOOD
            if getattr(req, "want_quiet_space", False) and quiet_space is True:
                score += W_QUIET
            if getattr(req, "want_less_crowded", False) and less_crowded is True:
                score += W_LESS_CROWD
            if getattr(req, "need_changing_station", False) and changing_station is True:
                score += W_CHANGING

            # Skips/penalties
            skipped = f.get("skipped_count") or 0
            try:
                skipped = float(skipped)
            except Exception:
                skipped = 0
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

    except HTTPException:
        # Bubble up clean client errors (400s)
        raise
    except Exception as e:
        # Log server-side; avoid leaking as 500 to the client
        print("Recommendation error:", repr(e))
        raise HTTPException(status_code=400, detail="Invalid input or dataset row; please check coordinates.")

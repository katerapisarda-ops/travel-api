from fastapi import FastAPI, HTTPException, Query, Request
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import os
import re, math, requests
from geopy.distance import geodesic
from fastapi.responses import JSONResponse 
import traceback
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
import time
from typing import Any, Dict, List, Tuple
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ------------------ Echo model for testing ------------------
class EchoBody(BaseModel):
    # Accept either "lon" or "lng" from the client
    lat: float
    lon: float = Field(..., alias="lng")
    time_available_mins: int | None = None

    # Allow extra fields (so you can see everything your app sends)
    model_config = ConfigDict(
        populate_by_name=True,  # lets "lon" work even if alias present
        extra="allow"           # keep unknown keys instead of rejecting them
    )

class RecReq(BaseModel):
    lat: float
    lng: float | None = None
    lon: float | None = None
    time: int = 90

class RecReq(BaseModel):
    lat: float
    lng: float | None = None
    lon: float | None = None
    time: int = 90
    debug: bool | None = False  # ← enable to get skip reasons back

def _now_ms() -> int:
    return int(time.time() * 1000)

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
    return {"pong": True}

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/recommendations")
def recommendations_get(
    lat: float = Query(...),
    lng: float | None = Query(None),
    lon: float | None = Query(None),
    time: int = Query(90),
    debug: bool | None = Query(False),
):
    return recommendations_post(RecReq(lat=lat, lng=lng, lon=lon, time=time, debug=debug))

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
    # Return a clean JSON instead of crashing the process
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error: {str(exc)}"}
    )

@app.get("/debug/vars")
def debug_vars():
    """Show which env vars are present (True/False only)."""
    return {
        "AIRTABLE_API_KEY_present": bool(os.getenv("AIRTABLE_API_KEY")),
        "AIRTABLE_BASE_ID_present": bool(os.getenv("AIRTABLE_BASE_ID")),
        "AIRTABLE_TABLE_present": bool(os.getenv("AIRTABLE_TABLE") or os.getenv("AIRTABLE_TABLE_NAME")),
        "AIRTABLE_VIEW_present": bool(os.getenv("AIRTABLE_VIEW")),
    }

@app.get("/debug/airtable")
def debug_airtable():
    """Try a small read from Airtable and return field names + a tiny sample."""
    try:
        from pyairtable import Table
    except Exception as e:
        return {"ok": False, "error": f"pyairtable not installed: {e}"}

    key = os.getenv("AIRTABLE_API_KEY")
    base = os.getenv("AIRTABLE_BASE_ID")
    table_name = os.getenv("AIRTABLE_TABLE") or os.getenv("AIRTABLE_TABLE_NAME")
    view = os.getenv("AIRTABLE_VIEW")  # optional

    if not (key and base and table_name):
        return {"ok": False, "error": "Missing Airtable env vars", "vars": {
            "key": bool(key), "base": bool(base), "table": bool(table_name), "view": bool(view)
        }}

    try:
        table = Table(key, base, table_name)
        recs = table.all(max_records=5, view=view)  # tiny read
        fields_list = [r.get("fields", {}) for r in recs]
        field_names = sorted({k for row in fields_list for k in row.keys()})
        return {"ok": True, "count": len(recs), "field_names": field_names, "sample": fields_list}
    except Exception as e:
        return {"ok": False, "error": f"Airtable fetch failed: {e.__class__.__name__}: {e}"}

@app.post("/echo")
async def echo(body: EchoBody):
    # model_dump() includes extra keys because of extra="allow"
    # by_alias=False ensures the key is "lon" in the response even if client sent "lng"
    return {"received": body.model_dump(by_alias=False)}

@app.post("/recommendations")
def recommendations_post(req: RecReq):
    t0 = _now_ms()
    longitude = req.lng if req.lng is not None else req.lon
    if longitude is None:
        raise HTTPException(status_code=400, detail="Provide either lng or lon")

    debug_rows: List[Dict[str, Any]] = []
    total_rows = 0
    kept: List[Dict[str, Any]] = []

    # 1) Fetch source data (Airtable/CSV/etc.)
    # TODO: replace with your real fetch; make sure it can't silently fail to []
    try:
        # Example:
        # rows = airtable_fetch()
        rows = []  # <-- YOUR REAL DATA HERE
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

    total_rows = len(rows)

    # 2) Iterate and apply your filters; collect reasons if skipped
    for r in rows:
        reasons: List[str] = []
        title = r.get("title") or r.get("name") or "Untitled"

        # Example geo filter
        lat = r.get("lat")
        # NOTE: your data uses `lon` not `lng`; keep that consistent:
        lon = r.get("lon") if r.get("lon") is not None else r.get("lng")

        if lat is None or lon is None:
            reasons.append("missing lat/lon")

        if reasons:
            if req.debug:
                debug_rows.append({"title": title, "skip": reasons})
            continue

        # Otherwise compute score and keep
        score = float(r.get("score") or 0)
        kept.append({
            "id": r.get("id") or title,
            "title": title,
            "score": score,
            "lat": lat,
            "lon": lon,
            "address": r.get("address"),
            "website": r.get("website"),
            # add any other fields you return to the app
        })

    # 3) Sort & cap
    kept.sort(key=lambda x: x.get("score", 0), reverse=True)
    recs = kept[:50]

    # 4) If empty, optionally return a “why it’s empty” summary
    dur_ms = _now_ms() - t0
    result = {
        "meta": {
            "total_rows": total_rows,
            "returned": len(recs),
            "duration_ms": dur_ms,
            "echo": {"lat": req.lat, "lng_or_lon": longitude, "time": req.time},
        },
        "recommendations": recs,
    }
    if req.debug:
        result["debug"] = debug_rows[:100]  # don’t explode the payload
    return result

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

def get_weather_condition(w: Optional[Dict[str, Any] | str]) -> Optional[str]:
    """
    Normalize req.weather into a simple lowercase condition string.
    Accepts:
      - "clouds" / "rain" / "clear" (string)
      - {"condition": "Clouds"} (simple dict)
      - OpenWeather: {"weather": [{"main": "Clouds"}], ...}
    Returns: e.g. "clouds", "rain", "clear", or None.
    """
    try:
        if w is None:
            return None
        if isinstance(w, str):
            c = w.strip().lower()
            return c or None
        if isinstance(w, dict):
            # simple shape
            val = w.get("condition")
            if isinstance(val, str) and val.strip():
                return val.strip().lower()

            # OpenWeather-like
            arr = w.get("weather")
            if isinstance(arr, list) and arr and isinstance(arr[0], dict):
                main = arr[0].get("main")
                if isinstance(main, str) and main.strip():
                    return main.strip().lower()
        return None
    except Exception:
        return None

def safe_bad_outdoor(w) -> bool:
    """
    Wrapper around your existing weather_bad_for_outdoor() that
    normalizes input and never throws.
    """
    try:
        condition = get_weather_condition(w)
        # If you already have weather_bad_for_outdoor(condition: str|None) defined:
        return bool(weather_bad_for_outdoor(condition))
    except Exception:
        # Minimal fallback heuristic (only used if your helper isn't available)
        c = get_weather_condition(w)
        if not c:
            return False
        # Treat clearly nasty conditions as "bad for outdoor"
        BAD = ("rain", "thunder", "storm", "snow", "sleet", "hail")
        if any(k in c for k in BAD):
            return True
        # Optional: very windy
        if "wind" in c and ("strong" in c or "gust" in c):
            return True
        return False

# ------------------ Airtable config ------------------
BASE_ID = "appApTB0N861wvwFU"
API_TOKEN = os.environ.get("AIRTABLE_API_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
TABLE_USER = "User Profiles"
TABLE_EXPERIENCES = "Experiences"

# ------------------ Request model ------------------
class RecRequest(BaseModel):
    lat: float
    lon: float = Field(..., alias="lng")
    user_id: Optional[str] = None
    interests: Optional[List[str]] = None
    vibes: Optional[List[str]] = None
    transport_mode: Optional[str] = None
    time_available_mins: Optional[int] = None

    # 👇 accept either a weather string ("clouds") or an object
    weather: Optional[Union[str, Dict[str, Any]]] = None

    time_of_day: Optional[str] = None

    # 👇 allow fractional ages like 2.5
    child_age_years: Optional[float] = None

    need_stroller_friendly: Optional[bool] = None
    want_food_nearby: Optional[bool] = None
    want_quiet_space: Optional[bool] = None
    want_less_crowded: Optional[bool] = None
    need_changing_station: Optional[bool] = None
    
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

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

def get_weather_condition(w) -> Optional[str]:
    """Accepts string ('clouds') or dict({'weather': [{'main': 'Clouds'}], ...}) and returns a lowercase condition."""
    try:
        if w is None:
            return None
        if isinstance(w, str):
            return w.strip().lower() or None
        if isinstance(w, dict):
            # common shapes
            if "condition" in w and isinstance(w["condition"], str):
                return w["condition"].strip().lower() or None
            # OpenWeather style
            arr = w.get("weather")
            if isinstance(arr, list) and arr and isinstance(arr[0], dict):
                main = arr[0].get("main")
                if isinstance(main, str):
                    return main.strip().lower() or None
        return None
    except Exception:
        return None

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
            
            bad_outdoor = safe_bad_outdoor(req.weather)  # ← normalized & guarded

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
                "lat": latlng[0],              # 👈 add
                "lon": latlng[1],              # 👈 add
                "address": f.get("address") or None,  # if you have it
                "website": f.get("website") or None,  # if you have it
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

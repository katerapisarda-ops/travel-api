# main.py
from __future__ import annotations

import os, re, time, requests
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from geopy.distance import geodesic


# -------------------- App & CORS --------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev-friendly; tighten for prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Env & Airtable config --------------------
AIRTABLE_KEY = (
    os.getenv("AIRTABLE_API_KEY")
    or os.getenv("AIRTABLE_PAT")
    or os.getenv("AIRTABLE_TOKEN")
)
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")

# Per-table identifiers (ID preferred, else name)
AIRTABLE_USER_IDENT = os.getenv("AIRTABLE_TABLE_USER_ID") or os.getenv("AIRTABLE_TABLE_USER_NAME")
AIRTABLE_EXP_IDENT  = os.getenv("AIRTABLE_TABLE_EXPERIENCES_ID") or os.getenv("AIRTABLE_TABLE_EXPERIENCES_NAME")
AIRTABLE_EVENT_SERIES_IDENT = os.getenv("AIRTABLE_TABLE_EVENT_SERIES_ID") or os.getenv("AIRTABLE_TABLE_EVENT_SERIES_NAME")
AIRTABLE_EVENT_OCCURRENCES_IDENT = os.getenv("AIRTABLE_TABLE_EVENT_OCCURRENCES_ID") or os.getenv("AIRTABLE_TABLE_EVENT_OCCURRENCES_NAME")

def _airtable_path_component(ident: str) -> str:
    """If ident looks like a table ID (tbl...), use as-is; otherwise URL-encode the name."""
    if not ident:
        return ""
    return ident if ident.startswith("tbl") else quote(ident, safe="")

def _fetch_airtable_records_by_ident(table_ident: str) -> List[Dict]:
    if not (AIRTABLE_KEY and AIRTABLE_BASE_ID and table_ident):
        raise HTTPException(status_code=500, detail="Airtable config missing (key/base/table).")
    path = _airtable_path_component(table_ident)
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{path}"
    headers = {"Authorization": f"Bearer {AIRTABLE_KEY}"}
    all_records: List[Dict] = []
    offset = None
    while True:
        params = {"offset": offset} if offset else {}
        res = requests.get(url, headers=headers, params=params, timeout=20)
        res.raise_for_status()
        data = res.json()
        all_records.extend(data.get("records", []))
        offset = data.get("offset")
        if not offset:
            break
    return all_records

# -------------------- Small utils --------------------
def _now_ms() -> int:
    return int(time.time() * 1000)

LATLNG_RE = re.compile(r'^\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*$')
SF_CENTER = (37.7793, -122.4193)
BAY_RADIUS_MI = 60.0

def parse_latlng(s: str) -> Optional[Tuple[float, float]]:
    m = LATLNG_RE.match(s) if isinstance(s, str) else None
    if not m:
        return None
    lat = float(m.group(1)); lng = float(m.group(2))
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return None
    return (lat, lng)

def safe_distance_miles(a: Tuple[float,float], b: Tuple[float,float]) -> Optional[float]:
    try:
        return geodesic(a, b).miles
    except Exception:
        return None

def is_within_bay_area(user_ll: Tuple[float,float]) -> bool:
    d = safe_distance_miles(user_ll, SF_CENTER)
    return (d is not None) and (d <= BAY_RADIUS_MI)

def parse_bool(value) -> Optional[bool]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"true","yes","y","1"}: return True
    if s in {"false","no","n","0"}: return False
    return None

def within_time(estimate: Optional[int], available: Optional[int]) -> bool:
    if not estimate or not available:
        return True
    return estimate <= available

def max_distance_for_transport(mode: str) -> float:
    return {
        "walking": 1.2,
        "transit": 3.5,
        "driving": 10.0,
    }.get((mode or "").lower(), 1.2)

def weather_bad_for_outdoor(weather: Optional[str]) -> bool:
    return (weather or "") in {"rain","drizzle","thunderstorm","fog"}

def get_weather_condition(w: Optional[Union[str, Dict[str, Any]]]) -> Optional[str]:
    try:
        if w is None:
            return None
        if isinstance(w, str):
            return w.strip().lower() or None
        if isinstance(w, dict):
            if isinstance(w.get("condition"), str):
                v = w["condition"].strip().lower()
                return v or None
            arr = w.get("weather")
            if isinstance(arr, list) and arr and isinstance(arr[0], dict):
                main = arr[0].get("main")
                if isinstance(main, str):
                    v = main.strip().lower()
                    return v or None
        return None
    except Exception:
        return None

def tod_fits(tod: Optional[str], hours_hint: Optional[str]) -> Optional[bool]:
    if not tod or not hours_hint:
        return None
    t = tod.strip().lower()
    h = str(hours_hint).strip().lower()
    if "morning" in h and t == "morning": return True
    if "midday"  in h and t == "midday":  return True
    if "afternoon" in h and t == "afternoon": return True
    if "evening" in h and t == "evening": return True
    return False

def safe_bad_outdoor(w) -> bool:
    try:
        return weather_bad_for_outdoor(get_weather_condition(w))
    except Exception:
        c = get_weather_condition(w)
        if not c: return False
        BAD = ("rain","thunder","storm","snow","sleet","hail")
        if any(k in c for k in BAD): return True
        if "wind" in c and ("strong" in c or "gust" in c): return True
        return False

def age_fits(age_range_str: Optional[str], child_age: Optional[float]) -> Optional[bool]:
    """Check if child age fits the age range"""
    if not age_range_str or child_age is None:
        return None
    
    age_range = age_range_str.lower()
    if "all ages" in age_range:
        return True
    elif "baby" in age_range and child_age <= 1:
        return True
    elif "toddler" in age_range and 1 < child_age <= 3:
        return True
    elif "preschool" in age_range and 3 < child_age <= 5:
        return True
    elif "older kid" in age_range and child_age > 6:
        return True
    
    return False

def calculate_event_status(event_start: str, event_end: str, event_date: str) -> str:
    """Calculate event status: happening_now, happening_soon, happening_today, happening_weekend"""
    from datetime import datetime, timedelta
    import pytz
    
    try:
        # Parse event datetime
        sf_tz = pytz.timezone('America/Los_Angeles')
        now = datetime.now(sf_tz)
        
        # Parse event date and times
        event_datetime_start = datetime.strptime(f"{event_date} {event_start}", "%Y-%m-%d %I:%M%p")
        event_datetime_end = datetime.strptime(f"{event_date} {event_end}", "%Y-%m-%d %I:%M%p")
        
        # Localize to SF timezone
        event_datetime_start = sf_tz.localize(event_datetime_start)
        event_datetime_end = sf_tz.localize(event_datetime_end)
        
        # Check if happening now
        if event_datetime_start <= now <= event_datetime_end:
            return "happening_now"
        
        # Check if happening soon (within next 2 hours)
        elif event_datetime_start > now and (event_datetime_start - now).total_seconds() <= 7200:
            return "happening_soon"
        
        # Check if happening today
        elif event_datetime_start.date() == now.date():
            return "happening_today"
        
        # Check if happening this weekend (Friday-Sunday)
        elif event_datetime_start.date() >= now.date() and event_datetime_start.weekday() >= 4:
            return "happening_weekend"
        
        return "upcoming"
        
    except Exception:
        return "upcoming"

def fetch_events_for_experience(experience_id: str) -> List[Dict]:
    """Fetch events linked to a specific experience"""
    try:
        if not (AIRTABLE_EVENT_SERIES_IDENT and AIRTABLE_EVENT_OCCURRENCES_IDENT):
            return []
            
        # Fetch event series linked to this experience
        series_records = _fetch_airtable_records_by_ident(AIRTABLE_EVENT_SERIES_IDENT)
        occurrence_records = _fetch_airtable_records_by_ident(AIRTABLE_EVENT_OCCURRENCES_IDENT)
        
        events = []
        for series in series_records:
            series_fields = series.get("fields", {})
            linked_venue = series_fields.get("linked_venue_experience", [])
            
            # Check if this series is linked to our experience
            if experience_id in linked_venue:
                # Find matching occurrences
                event_occurrences = series_fields.get("Event Occurrences", [])
                
                for occ_id in event_occurrences:
                    # Find the occurrence record
                    for occ in occurrence_records:
                        if occ.get("id") == occ_id:
                            occ_fields = occ.get("fields", {})
                            
                            event_data = {
                                "event_name": series_fields.get("event_name"),
                                "event_date": occ_fields.get("date"),
                                "start_time": series_fields.get("start_time"),
                                "end_time": series_fields.get("end_time"),
                                "event_description": series_fields.get("event_description"),
                                "parent_insider_tips": series_fields.get("parent_insider_tips"),
                                "event_type": series_fields.get("event_type"),
                                "event_cost": series_fields.get("event_cost"),
                                "tickets_required": series_fields.get("tickets_required"),
                                "tickets_url": series_fields.get("tickets_url"),
                                "event_website": series_fields.get("event_website"),
                                "interest_tags": series_fields.get("interest_tags", []),
                                "vibe_tags": series_fields.get("vibe_tags", []),
                                "target_age_groups": series_fields.get("target_age_groups", []),
                                "weather_sensitive": series_fields.get("weather_sensitive"),
                            }
                            
                            # Calculate event status
                            if event_data["event_date"] and event_data["start_time"] and event_data["end_time"]:
                                event_data["status"] = calculate_event_status(
                                    event_data["start_time"], 
                                    event_data["end_time"], 
                                    event_data["event_date"]
                                )
                            else:
                                event_data["status"] = "upcoming"
                            
                            events.append(event_data)
                            break
        
        return events
        
    except Exception:
        return []

# -------------------- Models --------------------
class EchoBody(BaseModel):
    lat: float
    lon: float = Field(..., alias="lng")
    time_available_mins: int | None = None
    model_config = ConfigDict(populate_by_name=True, extra="allow")

class RecReq(BaseModel):
    lat: float
    lng: float | None = None
    lon: float | None = None
    time: int = 90
    interests: List[str] | None = None
    vibes: List[str] | None = None
    transport_mode: str | None = None
    weather: Any | None = None
    time_of_day: str | None = None
    child_age_years: float | None = None
    include_events: bool = False

# -------------------- Recommendation engine --------------------
def build_recommendations(
    *,
    user_lat: float,
    user_lon: float,
    time_available_mins: int | None = None,
    interests: List[str] | None = None,
    vibes: List[str] | None = None,
    transport_mode: str | None = None,
    weather: str | Dict | None = None,
    time_of_day: str | None = None,
    child_age_years: float | None = None,
    include_events: bool = False,
) -> List[Dict]:
    exp_rows = _fetch_airtable_records_by_ident(AIRTABLE_EXP_IDENT)

    # Debug logging for all records
    print("\nDEBUG: Airtable Records Raw Data:")
    for rec in exp_rows:
        f = rec.get("fields", {})
        if "Noe" in str(f.get("title", "")):
            print(f"\nFound record with 'Noe' in title: {f.get('title')}")
            print(f"Record ID: {rec.get('id')}")
            print(f"Raw fields: {f}")
            print(f"Description field: {repr(f.get('description'))}")
            print(f"Parent Tips field: {repr(f.get('parent_insider_tips'))}")
            print("Field keys present:", list(f.keys()))

    # helper to coerce best_age_range -> single string
    def _first_str(x):
        if isinstance(x, list) and x:
            return str(x[0])
        if isinstance(x, str):
            return x
        return None

    interests_set = set((interests or []))
    vibes_set = set((vibes or []))
    max_dist_mi = max_distance_for_transport(transport_mode or "")
    user_loc = (float(user_lat), float(user_lon))
    if not is_within_bay_area(user_loc):
        user_loc = SF_CENTER

    bad_outdoor = safe_bad_outdoor(weather)

    scored: List[Dict] = []
    for rec in exp_rows:
        f = rec.get("fields", {})
        rid = rec.get("id")
        title = f.get("title", "Untitled")

        # coords: prefer split fields; fallback to "lat_lng"
        lat = f.get("lat")
        lon = f.get("lon") if f.get("lon") is not None else f.get("lng")
        if (lat is None or lon is None):
            parsed = parse_latlng(f.get("lat_lng_clean") or f.get("lat_lng") or "")
            if parsed:
                lat, lon = parsed

        if (lat is None or lon is None):
            continue

        # optional filters
        if f.get("open_now") is False:
            continue

        estimate = f.get("time_estimate_mins") or f.get("estimated_time_mins") or 0
        if time_available_mins is not None and not within_time(estimate, time_available_mins):
            continue

        weather_sensitive = (f.get("weather_sensitive") or "").strip().lower()
        if weather_sensitive in {"avoid_rain", "outdoor_only"} and bad_outdoor:
            continue

        dist_mi = safe_distance_miles(user_loc, (float(lat), float(lon)))
        if dist_mi is None or dist_mi > max_dist_mi:
            continue

        # scoring
        W_INTEREST, W_VIBE, W_DISTANCE = 3.0, 1.5, 2.5
        W_TOD, W_WEATHER, W_AGE = 0.75, 1.5, 2.0
        W_STROLLER, W_FOOD, W_QUIET, W_LESS_CROWD, W_CHANGING = 1.2, 1.3, 1.1, 1.1, 1.4
        W_SKIP = 0.4

        score = 0.0

        interest_tags = set([x.lower() for x in f.get("interest_tags", [])])
        vibe_tags = set([x.lower() for x in f.get("vibe_tags", [])])

        score += W_INTEREST * len(interests_set & interest_tags)
        score += W_VIBE * len(vibes_set & vibe_tags)

        if max_dist_mi > 0:
            proximity = max(0.0, 1.0 - (dist_mi / max_dist_mi))
            score += W_DISTANCE * proximity

        tod_hint = f.get("best_time_of_day")
        tod_fit = tod_fits(time_of_day, tod_hint)
        if tod_fit is True:
            score += W_TOD
        elif tod_fit is False:
            score -= W_TOD * 0.5

        is_indoorish = bool(parse_bool(f.get("has_quiet_space")))
        if bad_outdoor and is_indoorish:
            score += W_WEATHER

        fits_age = None
        try:
            fits_age = age_fits(f.get("best_age_range"), child_age_years)
        except Exception:
            pass
        if fits_age is True:
            score += W_AGE
        elif fits_age is False:
            score -= W_AGE * 0.75

        stroller_friendly = parse_bool(f.get("stroller_friendly"))
        food_nearby       = parse_bool(f.get("food_nearby"))
        quiet_space       = parse_bool(f.get("has_quiet_space"))
        less_crowded      = parse_bool(f.get("less_crowded_place"))
        changing_station  = parse_bool(f.get("has_changing_station"))
        outdoor_space     = parse_bool(f.get("has_outdoor_space"))

        skipped = float(f.get("skipped_count") or 0)
        score -= W_SKIP * skipped

        # Fetch events if requested
        events = []
        if include_events:
            events = fetch_events_for_experience(rid)

        scored.append({
            "id": rid,  # 👈 include Airtable record id
            "title": title,
            "score": round(score, 2),
            "distance_miles": round(dist_mi, 2),
            "lat": float(lat),
            "lon": float(lon),
            "address": f.get("address") or None,
            "website": f.get("website") or None,
            "cost_tier": f.get("cost_tier") or None,  # "$", "$$", "Free"
            "time_estimate_mins": estimate or None,
            "weather_sensitive": weather_sensitive or None,
            "best_age_range": _first_str(f.get("best_age_range")),  # 👈 normalize
            "interest_tags": list(interest_tags) if interest_tags else [],
            "vibe_tags": list(vibe_tags) if vibe_tags else [],
            "restrooms_nearby": parse_bool(f.get("restrooms_nearby")),
            "has_changing_station": changing_station,
            "food_nearby": food_nearby,
            "stroller_friendly": stroller_friendly,
            "has_playground_nearby": parse_bool(f.get("has_playground_nearby")),
            "has_outdoor_space": outdoor_space,
            "less_crowded_place": less_crowded,
            "description": f.get("description") or None,
            "parent_insider_tips": f.get("parent_insider_tips") or None,
            "neighborhood_original": f.get("neighborhood_original") or None,
            "Google Rating": f.get("Google Rating") or None,
            "title_sub_tags": f.get("title_sub_tags") or None,
            "image_url": (f.get("Photo URL") or f.get("image_url") or f.get("photo") or None),
            "events": events if include_events else None,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    
    # Add before return scored
    print("\nDEBUG: Final recommendations:")
    for rec in scored[:5]:  # Just look at top 5 for brevity
        if "Noe" in rec.get("title", ""):
            print(f"\nProcessed recommendation for: {rec.get('title')}")
            print(f"Description: {repr(rec.get('description'))}")
            print(f"Parent Tips: {repr(rec.get('parent_insider_tips'))}")

    return scored

# -------------------- Routes --------------------
@app.get("/")
def root(): return {"ok": True, "hint": "Try /ping and /docs"}

@app.get("/ping")
def ping(): return {"pong": True}

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.get("/debug/vars")
def debug_vars():
    return {
        "AIRTABLE_API_KEY_present": bool(AIRTABLE_KEY),
        "AIRTABLE_BASE_ID_present": bool(AIRTABLE_BASE_ID),
        "user_ident": AIRTABLE_USER_IDENT,
        "user_uses_id": bool(AIRTABLE_USER_IDENT and AIRTABLE_USER_IDENT.startswith("tbl")),
        "exp_ident": AIRTABLE_EXP_IDENT,
        "exp_uses_id": bool(AIRTABLE_EXP_IDENT and AIRTABLE_EXP_IDENT.startswith("tbl")),
        "event_series_ident": AIRTABLE_EVENT_SERIES_IDENT,
        "event_series_uses_id": bool(AIRTABLE_EVENT_SERIES_IDENT and AIRTABLE_EVENT_SERIES_IDENT.startswith("tbl")),
        "event_occurrences_ident": AIRTABLE_EVENT_OCCURRENCES_IDENT,
        "event_occurrences_uses_id": bool(AIRTABLE_EVENT_OCCURRENCES_IDENT and AIRTABLE_EVENT_OCCURRENCES_IDENT.startswith("tbl")),
    }

@app.get("/debug/airtable")
def debug_airtable():
    try:
        user_ok = None
        if AIRTABLE_USER_IDENT:
            _ = _fetch_airtable_records_by_ident(AIRTABLE_USER_IDENT)[:1]
            user_ok = True
        _ = _fetch_airtable_records_by_ident(AIRTABLE_EXP_IDENT)[:1]
        exp_ok = True
        return {"ok": True, "user_ok": user_ok, "exp_ok": True,
                "user_uses_id": bool(AIRTABLE_USER_IDENT and AIRTABLE_USER_IDENT.startswith("tbl")),
                "exp_uses_id": bool(AIRTABLE_EXP_IDENT and AIRTABLE_EXP_IDENT.startswith("tbl"))}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

@app.exception_handler(Exception)
async def catch_all_exceptions(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {str(exc)}"})

# Simple echo for testing payload shape
@app.post("/echo")
async def echo(body: EchoBody):
    return {"received": body.model_dump(by_alias=False)}

# GET shim (query params)
@app.get("/recommendations")
def recommendations_get(
    lat: float = Query(...),
    lng: float | None = Query(None),
    lon: float | None = Query(None),
    time: int = Query(90),
    include_events: bool = Query(False),
):
    longitude = lng if lng is not None else lon
    if longitude is None:
        raise HTTPException(status_code=400, detail="Provide either lng or lon")
    t0 = _now_ms()
    recs = build_recommendations(user_lat=lat, user_lon=longitude, time_available_mins=time, include_events=include_events)
    dur = _now_ms() - t0
    return {"meta": {"total_rows": len(recs), "returned": min(len(recs), 50), "duration_ms": dur,
                     "echo": {"lat": lat, "lng_or_lon": longitude, "time": time, "include_events": include_events}},
            "recommendations": recs[:50]}

# POST (body with richer options)
@app.post("/recommendations")
def recommendations_post(req: RecReq):
    longitude = req.lng if req.lng is not None else req.lon
    if longitude is None:
        raise HTTPException(status_code=400, detail="Provide either lng or lon")
    t0 = _now_ms()
    recs = build_recommendations(
        user_lat=req.lat,
        user_lon=longitude,
        time_available_mins=req.time,
        interests=req.interests or [],
        vibes=req.vibes or [],
        transport_mode=req.transport_mode,
        weather=req.weather,
        time_of_day=req.time_of_day,
        child_age_years=req.child_age_years,
        include_events=req.include_events,
    )
    dur = _now_ms() - t0
    return {"meta": {"total_rows": len(recs), "returned": min(len(recs), 50), "duration_ms": dur,
                     "echo": {"lat": req.lat, "lng_or_lon": longitude, "time": req.time}},
            "recommendations": recs[:50]}
# -------------------- End of file --------------------

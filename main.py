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
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


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

def calculate_event_status(event_start: str, event_end: str, event_date: str, soon_window_hours: int = 2) -> str:
    """Calculate event status with sane semantics:
    - past: event already ended
    - happening_now: currently in progress
    - happening_soon: starts within 2 hours
    - happening_today: later today
    - happening_weekend: falls in the current/upcoming weekend window (Fri-Sun of this week)
    - upcoming: any other future date
    """
    from datetime import datetime, timedelta
    import pytz

    try:
        # Timezone: use SF local time by default (dataset is Bay Area)
        sf_tz = pytz.timezone('America/Los_Angeles')
        now = datetime.now(sf_tz)

        # Parse event date and times (12-hour clock with AM/PM)
        event_datetime_start = datetime.strptime(f"{event_date} {event_start}", "%Y-%m-%d %I:%M%p")
        event_datetime_end = datetime.strptime(f"{event_date} {event_end}", "%Y-%m-%d %I:%M%p")

        # Localize
        event_datetime_start = sf_tz.localize(event_datetime_start)
        event_datetime_end = sf_tz.localize(event_datetime_end)

        # Past check first
        if event_datetime_end < now:
            return "past"

        # Now
        if event_datetime_start <= now <= event_datetime_end:
            return "happening_now"

        # Soon (within configured window)
        if event_datetime_start > now and (event_datetime_start - now).total_seconds() <= max(0, soon_window_hours) * 3600:
            return "happening_soon"

        # Today
        if event_datetime_start.date() == now.date():
            return "happening_today"

        # This weekend (Fri-Sun window for this week only)
        # Determine Friday and Sunday of "this" weekend relative to now
        # If Mon-Thu, this refers to the upcoming Fri-Sun; if Fri-Sun, it's the current Fri-Sun span.
        wd = now.weekday()  # Mon=0 ... Sun=6
        if wd <= 3:  # Mon-Thu -> upcoming weekend
            friday = (now + timedelta(days=(4 - wd))).date()
        else:  # Fri-Sun -> current weekend's Friday
            friday = (now - timedelta(days=(wd - 4))).date()
        sunday = friday + timedelta(days=2)

        ev_date = event_datetime_start.date()
        if friday <= ev_date <= sunday:
            return "happening_weekend"

        # Future but not this weekend
        return "upcoming"

    except Exception:
        # On parse error, default to upcoming (non-breaking)
        return "upcoming"

def fetch_events_for_experience(experience_id: str, *, soon_window_hours: int = 2) -> List[Dict]:
    """Fetch events linked to a specific experience (robust to field-name differences).

    Heuristics used:
    - Detect the series->experience link either via known field 'linked_venue_experience' or any list field containing the experience_id.
    - Detect series->occurrences either via known field 'Event Occurrences' or any list field of rec IDs that exist in the occurrences table.
    - Read occurrence date from common variants: 'date', 'Date', 'event_date', 'occurrence_date'.
    - Read start/end time from common variants on the series: start_time|Start Time|start, end_time|End Time|end.
    """
    try:
        if not (AIRTABLE_EVENT_SERIES_IDENT and AIRTABLE_EVENT_OCCURRENCES_IDENT):
            return []

        series_records = _fetch_airtable_records_by_ident(AIRTABLE_EVENT_SERIES_IDENT)
        occurrence_records = _fetch_airtable_records_by_ident(AIRTABLE_EVENT_OCCURRENCES_IDENT)

        occ_by_id = {o.get("id"): (o.get("fields", {})) for o in occurrence_records}
        occ_id_set = set(occ_by_id.keys())

        def pick_field(d: Dict, candidates: List[str]) -> Any:
            # Case-insensitive, space/underscore-insensitive pick
            if not isinstance(d, dict):
                return None
            norm_map = {"".join(k.lower().replace("_", " ").split()): k for k in d.keys()}
            for c in candidates:
                key_norm = "".join(c.lower().replace("_", " ").split())
                if key_norm in norm_map:
                    return d.get(norm_map[key_norm])
            return None

        def get_occurrence_date(of: Dict) -> Any:
            # Prefer start_datetime's date if present, else common variants
            sd = of.get("start_datetime")
            if isinstance(sd, str) and "T" in sd:
                return sd.split("T")[0]
            for k in ["date", "Date", "event_date", "occurrence_date"]:
                if k in of and of[k]:
                    return of[k]
            return None

        def get_times_from_occurrence(of: Dict) -> tuple[Any, Any]:
            """Return (start_time_str, end_time_str) in '%I:%M%p' if possible from start/end datetime."""
            from datetime import datetime
            def fmt_time(s: str | None) -> Any:
                if not isinstance(s, str):
                    return None
                t = s.replace("Z", "+00:00")
                try:
                    dt = datetime.fromisoformat(t)
                    # format to 12-hour with AM/PM
                    return dt.strftime("%I:%M%p").lstrip("0")
                except Exception:
                    return None
            return fmt_time(of.get("start_datetime")), fmt_time(of.get("end_datetime"))

        def get_series_time(sf: Dict, which: str) -> Any:
            if which == "start":
                return pick_field(sf, ["start_time", "Start Time", "start"])
            else:
                return pick_field(sf, ["end_time", "End Time", "end"])

        events: List[Dict] = []

        for series in series_records:
            series_fields = series.get("fields", {})

            # Determine if this series links to the target experience
            linked_ok = False
            # 1) Known field name path
            linked_list = series_fields.get("linked_venue_experience")
            if isinstance(linked_list, list) and experience_id in linked_list:
                linked_ok = True
            else:
                # 2) Heuristic: any list field containing the experience id
                for v in series_fields.values():
                    if isinstance(v, list) and experience_id in v:
                        linked_ok = True
                        break
            if not linked_ok:
                continue

            # Collect occurrence IDs for this series
            occ_ids: List[str] = []
            # Preferred field name
            preferred = series_fields.get("Event Occurrences")
            if isinstance(preferred, list) and all(isinstance(x, str) for x in preferred):
                occ_ids = [x for x in preferred if x in occ_id_set]
            # Heuristic fallback: any list field that is a subset of occurrence ids
            if not occ_ids:
                for v in series_fields.values():
                    if isinstance(v, list) and v and all(isinstance(x, str) and x.startswith("rec") for x in v):
                        subset = [x for x in v if x in occ_id_set]
                        if subset:
                            occ_ids = subset
                            break
            # Ultimate fallback: derive by scanning occurrences for a link back to this series id
            if not occ_ids:
                sid = series.get("id")
                for oid, of in occ_by_id.items():
                    for val in of.values():
                        if isinstance(val, list) and sid in val:
                            occ_ids.append(oid)

            # Build event entries
            for occ_id in occ_ids:
                of = occ_by_id.get(occ_id, {})

                event_date = get_occurrence_date(of)
                # Prefer precise times from occurrence datetimes; fallback to series-level times
                occ_start_time, occ_end_time = get_times_from_occurrence(of)
                start_time = occ_start_time or get_series_time(series_fields, "start")
                end_time = occ_end_time or get_series_time(series_fields, "end")

                event_name = pick_field(series_fields, ["event_name", "Event Name", "title"]) or series_fields.get("title")

                event_data = {
                    "event_name": event_name,
                    "event_date": event_date,
                    "start_time": start_time,
                    "end_time": end_time,
                    "event_description": pick_field(series_fields, ["event_description", "description", "Event Description"]) or None,
                    "parent_insider_tips": series_fields.get("parent_insider_tips"),
                    "event_type": series_fields.get("event_type"),
                    "event_cost": series_fields.get("event_cost"),
                    "tickets_required": series_fields.get("tickets_required"),
                    "tickets_url": series_fields.get("tickets_url"),
                    "event_website": series_fields.get("event_website") or series_fields.get("website"),
                    "interest_tags": series_fields.get("interest_tags", []) or [],
                    "vibe_tags": series_fields.get("vibe_tags", []) or [],
                    "target_age_groups": series_fields.get("target_age_groups", []) or [],
                    "weather_sensitive": series_fields.get("weather_sensitive"),
                }

                if event_date and start_time and end_time:
                    try:
                        event_data["status"] = calculate_event_status(start_time, end_time, event_date, soon_window_hours=soon_window_hours)
                    except Exception:
                        event_data["status"] = "upcoming"
                else:
                    event_data["status"] = "upcoming"

                events.append(event_data)

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
    event_timeframe: str | None = None  # one of: now, soon, today, weekend, upcoming, all
    soon_hours: int = 2
    drop_past_events: bool = True

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
    event_timeframe: str | None = None,
    soon_hours: int = 2,
    drop_past_events: bool = True,
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
            events = fetch_events_for_experience(rid, soon_window_hours=soon_hours)

            # Drop past events if requested
            if drop_past_events:
                events = [e for e in events if str(e.get("status")) != "past"]

            # Filter by timeframe if provided
            tf = (event_timeframe or "").strip().lower() if event_timeframe else None
            if tf and tf != "all":
                allowed = None
                if tf in {"now", "happening_now"}:
                    allowed = {"happening_now"}
                elif tf in {"soon", "happening_soon"}:
                    allowed = {"happening_soon"}
                elif tf == "today":
                    allowed = {"happening_today"}
                elif tf == "weekend":
                    allowed = {"happening_weekend"}
                elif tf in {"upcoming", "future"}:
                    allowed = {"happening_today", "happening_weekend", "upcoming"}

                if allowed is not None:
                    events = [e for e in events if str(e.get("status")) in allowed]

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

@app.get("/debug/airtable_events")
def debug_airtable_events(limit: int = Query(3, ge=1, le=20)):
    """Summarize Event Series and Occurrence schemas and sample field keys for troubleshooting."""
    try:
        series = _fetch_airtable_records_by_ident(AIRTABLE_EVENT_SERIES_IDENT)[:limit]
        occs = _fetch_airtable_records_by_ident(AIRTABLE_EVENT_OCCURRENCES_IDENT)[:limit]
        series_summary = []
        for s in series:
            f = s.get("fields", {})
            series_summary.append({
                "id": s.get("id"),
                "title": f.get("title") or f.get("event_name"),
                "field_keys": sorted(list(f.keys())),
                "linked_venue_field_present": "linked_venue_experience" in f,
                "has_event_occurrences_field": isinstance(f.get("Event Occurrences"), list),
            })
        occ_summary = []
        for o in occs:
            f = o.get("fields", {})
            occ_summary.append({
                "id": o.get("id"),
                "date_like": f.get("date") or f.get("Date") or f.get("event_date") or f.get("occurrence_date"),
                "field_keys": sorted(list(f.keys())),
            })
        return {
            "event_series_ident": AIRTABLE_EVENT_SERIES_IDENT,
            "event_occurrences_ident": AIRTABLE_EVENT_OCCURRENCES_IDENT,
            "series_samples": series_summary,
            "occurrence_samples": occ_summary,
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

@app.get("/debug/search_experiences")
def debug_search_experiences(title_contains: str = Query(..., description="Case-insensitive substring to search in experience titles")):
    """Search Experiences by title substring to discover record IDs for targeted debugging."""
    try:
        matches = []
        exp_rows = _fetch_airtable_records_by_ident(AIRTABLE_EXP_IDENT)
        needle = (title_contains or "").strip().lower()
        for rec in exp_rows:
            f = rec.get("fields", {})
            title = str(f.get("title", ""))
            if needle in title.lower():
                lat = f.get("lat")
                lon = f.get("lon") if f.get("lon") is not None else f.get("lng")
                parsed = None
                if lat is None or lon is None:
                    parsed = parse_latlng(f.get("lat_lng_clean") or f.get("lat_lng") or f.get("display_lat_lng") or "")
                matches.append({
                    "id": rec.get("id"),
                    "title": title,
                    "lat": lat,
                    "lon": lon,
                    "lat_lng_raw": f.get("lat_lng") or f.get("lat_lng_clean") or f.get("display_lat_lng") or None,
                    "parsed_lat": parsed[0] if parsed else None,
                    "parsed_lon": parsed[1] if parsed else None,
                    "address": f.get("address") or None,
                })
        return {"query": title_contains, "count": len(matches), "matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search error: {type(e).__name__}: {e}")

@app.get("/debug/events_for_experience")
def debug_events_for_experience(
    experience_id: str | None = Query(None, description="Airtable record id for the Experience (rec...)"),
    title_contains: str | None = Query(None, description="If id unknown, find Experience by case-insensitive title substring"),
    status: str | None = Query(None, description="Optional filter by status (e.g., happening_weekend,happening_today) comma-separated"),
    on_date: str | None = Query(None, description="Optional YYYY-MM-DD filter to only include events on this date"),
    soon_hours: int = Query(2, description="Window in hours to consider 'happening_soon'"),
):
    """Fetch linked events for a specific Experience by id or fuzzy title match.

    Useful to validate Airtable linkage for a known venue (e.g., Noe Valley Farmers Market).
    """
    try:
        chosen_id = experience_id
        chosen_title = None
        matched_by = "id" if experience_id else None

        if not chosen_id and title_contains:
            # find first match by title
            exp_rows = _fetch_airtable_records_by_ident(AIRTABLE_EXP_IDENT)
            needle = title_contains.strip().lower()
            candidates = []
            for rec in exp_rows:
                f = rec.get("fields", {})
                title = str(f.get("title", ""))
                if needle in title.lower():
                    candidates.append((rec.get("id"), title))
            if not candidates:
                raise HTTPException(status_code=404, detail=f"No Experience title contains '{title_contains}'")
            if len(candidates) > 1:
                # Return options to avoid picking the wrong one
                return {"multiple_matches": True, "options": [{"id": cid, "title": t} for cid, t in candidates]}
            chosen_id, chosen_title = candidates[0]
            matched_by = "title_contains"

        if not chosen_id:
            raise HTTPException(status_code=400, detail="Provide either experience_id or title_contains")

        # Build debug series info showing which series link to this experience
        series_info: List[Dict] = []
        try:
            series_records = _fetch_airtable_records_by_ident(AIRTABLE_EVENT_SERIES_IDENT)
            for s in series_records:
                sf = s.get("fields", {})
                linked = sf.get("linked_venue_experience")
                linked_ok = isinstance(linked, list) and (chosen_id in linked)
                if not linked_ok:
                    # heuristic: any list field contains chosen_id
                    for v in sf.values():
                        if isinstance(v, list) and chosen_id in v:
                            linked_ok = True
                            break
                if linked_ok:
                    occs = sf.get("Event Occurrences")
                    occ_count = len(occs) if isinstance(occs, list) else 0
                    series_info.append({
                        "series_id": s.get("id"),
                        "series_title": sf.get("title") or sf.get("event_name"),
                        "occurrence_count": occ_count,
                        "has_event_occurrences_field": isinstance(occs, list),
                    })
        except Exception:
            pass

        events = fetch_events_for_experience(chosen_id, soon_window_hours=soon_hours)

        # Optional filters
        if on_date:
            events = [e for e in events if str(e.get("event_date")) == on_date]
        if status:
            allowed = {s.strip().lower() for s in status.split(",")}
            events = [e for e in events if str(e.get("status", "")).lower() in allowed]

        return {
            "experience_id": chosen_id,
            "experience_title": chosen_title,
            "matched_by": matched_by,
            "series_linked": series_info,
            "events_total": len(events),
            "events": events,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"events debug error: {type(e).__name__}: {e}")

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
    event_timeframe: str | None = Query(None, description="Filter events: now, soon, today, weekend, upcoming, all"),
    soon_hours: int = Query(2, description="Window for 'soon' in hours"),
    drop_past_events: bool = Query(True, description="If true, omit events marked as 'past'"),
):
    longitude = lng if lng is not None else lon
    if longitude is None:
        raise HTTPException(status_code=400, detail="Provide either lng or lon")
    t0 = _now_ms()
    recs = build_recommendations(user_lat=lat, user_lon=longitude, time_available_mins=time, include_events=include_events,
                                 event_timeframe=event_timeframe, soon_hours=soon_hours, drop_past_events=drop_past_events)
    dur = _now_ms() - t0
    return {"meta": {"total_rows": len(recs), "returned": min(len(recs), 50), "duration_ms": dur,
                     "echo": {"lat": lat, "lng_or_lon": longitude, "time": time, "include_events": include_events,
                               "event_timeframe": event_timeframe, "soon_hours": soon_hours, "drop_past_events": drop_past_events}},
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
        event_timeframe=req.event_timeframe,
        soon_hours=req.soon_hours,
        drop_past_events=req.drop_past_events,
    )
    dur = _now_ms() - t0
    return {"meta": {"total_rows": len(recs), "returned": min(len(recs), 50), "duration_ms": dur,
                     "echo": {"lat": req.lat, "lng_or_lon": longitude, "time": req.time,
                               "event_timeframe": req.event_timeframe, "soon_hours": req.soon_hours,
                               "drop_past_events": req.drop_past_events}},
            "recommendations": recs[:50]}
# -------------------- End of file --------------------

"""
Client for the National Weather Service API (https://api.weather.gov).

The NWS API needs no key and no auth plumbing, so this module is purely about
turning locations into documents:

    "Chicago, IL"  ->  (41.8781, -87.6298)          resolve_location()
                   ->  grid point LOT/75,73          resolve_grid_point()
                   ->  active alerts + 7-day forecast
                   ->  normalized document records   fetch_documents()

Every document that comes out of :meth:`WeatherClient.fetch_documents` has the
same shape regardless of whether it started life as an alert or a forecast
period, which is what lets one embedding pipeline and one retrieval endpoint
serve both.

The NWS terms of service ask every client to identify itself in the
``User-Agent`` header with a contact address. Set ``NWS_USER_AGENT`` to
something like ``"weather-intelligence-app (you@example.com)"``.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Iterable

import requests

NWS_API_BASE_URL = os.environ.get("NWS_API_BASE_URL", "https://api.weather.gov")

DEFAULT_USER_AGENT = os.environ.get(
    "NWS_USER_AGENT", "weather-intelligence-app (contact: set NWS_USER_AGENT)"
)

# The NWS asks for "a few requests per second" at most. One resolve + one
# alerts + one forecast call per location is already modest, but a small
# delay keeps bulk syncs comfortably inside the guidance.
DEFAULT_REQUEST_DELAY_SECONDS = float(os.environ.get("NWS_REQUEST_DELAY", "0.2"))
DEFAULT_TIMEOUT = 30

SOURCE_TYPE_ALERT = "alert"
SOURCE_TYPE_FORECAST = "forecast"

# A "City, ST" string has no meaning to the NWS API, which only speaks
# latitude/longitude. This table covers the metro areas most likely to be
# used in a demo so the pipeline runs with zero external geocoding. Anything
# not listed here falls back to the free U.S. Census geocoder, and raw
# "lat,lon" input always works.
US_CITY_COORDINATES: dict[str, tuple[float, float]] = {
    "albuquerque, nm": (35.0844, -106.6504),
    "anchorage, ak": (61.2181, -149.9003),
    "atlanta, ga": (33.7490, -84.3880),
    "austin, tx": (30.2672, -97.7431),
    "baltimore, md": (39.2904, -76.6122),
    "billings, mt": (45.7833, -108.5007),
    "birmingham, al": (33.5186, -86.8104),
    "boise, id": (43.6150, -116.2023),
    "boston, ma": (42.3601, -71.0589),
    "buffalo, ny": (42.8864, -78.8784),
    "burlington, vt": (44.4759, -73.2121),
    "charleston, sc": (32.7765, -79.9311),
    "charlotte, nc": (35.2271, -80.8431),
    "cheyenne, wy": (41.1400, -104.8202),
    "chicago, il": (41.8781, -87.6298),
    "cincinnati, oh": (39.1031, -84.5120),
    "cleveland, oh": (41.4993, -81.6944),
    "columbus, oh": (39.9612, -82.9988),
    "dallas, tx": (32.7767, -96.7970),
    "denver, co": (39.7392, -104.9903),
    "des moines, ia": (41.5868, -93.6250),
    "detroit, mi": (42.3314, -83.0458),
    "fargo, nd": (46.8772, -96.7898),
    "hartford, ct": (41.7658, -72.6734),
    "honolulu, hi": (21.3069, -157.8583),
    "houston, tx": (29.7604, -95.3698),
    "indianapolis, in": (39.7684, -86.1581),
    "jackson, ms": (32.2988, -90.1848),
    "jacksonville, fl": (30.3322, -81.6557),
    "kansas city, mo": (39.0997, -94.5786),
    "las vegas, nv": (36.1699, -115.1398),
    "little rock, ar": (34.7465, -92.2896),
    "los angeles, ca": (34.0522, -118.2437),
    "louisville, ky": (38.2527, -85.7585),
    "memphis, tn": (35.1495, -90.0490),
    "miami, fl": (25.7617, -80.1918),
    "milwaukee, wi": (43.0389, -87.9065),
    "minneapolis, mn": (44.9778, -93.2650),
    "nashville, tn": (36.1627, -86.7816),
    "new orleans, la": (29.9511, -90.0715),
    "new york, ny": (40.7128, -74.0060),
    "norfolk, va": (36.8508, -76.2859),
    "oklahoma city, ok": (35.4676, -97.5164),
    "omaha, ne": (41.2565, -95.9345),
    "orlando, fl": (28.5383, -81.3792),
    "philadelphia, pa": (39.9526, -75.1652),
    "phoenix, az": (33.4484, -112.0740),
    "pittsburgh, pa": (40.4406, -79.9959),
    "portland, me": (43.6591, -70.2568),
    "portland, or": (45.5152, -122.6784),
    "providence, ri": (41.8240, -71.4128),
    "raleigh, nc": (35.7796, -78.6382),
    "reno, nv": (39.5296, -119.8138),
    "richmond, va": (37.5407, -77.4360),
    "sacramento, ca": (38.5816, -121.4944),
    "salt lake city, ut": (40.7608, -111.8910),
    "san antonio, tx": (29.4241, -98.4936),
    "san diego, ca": (32.7157, -117.1611),
    "san francisco, ca": (37.7749, -122.4194),
    "seattle, wa": (47.6062, -122.3321),
    "sioux falls, sd": (43.5460, -96.7313),
    "spokane, wa": (47.6588, -117.4260),
    "st. louis, mo": (38.6270, -90.1994),
    "tampa, fl": (27.9506, -82.4572),
    "tulsa, ok": (36.1540, -95.9928),
    "washington, dc": (38.9072, -77.0369),
    "wichita, ks": (37.6872, -97.3301),
}

_LAT_LON_RE = re.compile(
    r"^\s*(-?\d{1,3}(?:\.\d+)?)\s*[,/ ]\s*(-?\d{1,3}(?:\.\d+)?)\s*$"
)

_CENSUS_GEOCODER_URL = (
    "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
)


class LocationResolutionError(ValueError):
    """Raised when a location string cannot be turned into coordinates."""


def content_hash(text: str) -> str:
    """Stable hash of a narrative, used to detect re-issued/updated text.

    Alerts get updated and forecasts get re-issued several times a day. The
    hash lets the embedding job re-embed a document only when its text has
    actually changed, instead of every time it is re-synced.
    """
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _parse_timestamp(value: str | None) -> str | None:
    """Normalize an NWS ISO-8601 timestamp to UTC, or return None."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


class WeatherClient:
    """Thin, retry-friendly wrapper around the National Weather Service API."""

    def __init__(
        self,
        base_url: str | None = None,
        user_agent: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        request_delay: float = DEFAULT_REQUEST_DELAY_SECONDS,
    ):
        self.base_url = (base_url or NWS_API_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.request_delay = request_delay
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": user_agent or DEFAULT_USER_AGENT,
                "Accept": "application/geo+json",
            }
        )
        self._last_request_at = 0.0

    # -- HTTP ------------------------------------------------------------

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET a path on the NWS API and return the decoded JSON body."""
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

        resp = self._session.get(
            f"{self.base_url}{path}", params=params, timeout=self.timeout
        )
        self._last_request_at = time.monotonic()
        resp.raise_for_status()
        return resp.json()

    # -- Location resolution ---------------------------------------------

    def resolve_location(self, location: str) -> tuple[float, float]:
        """Turn a location string into ``(latitude, longitude)``.

        Accepts a raw ``"41.88,-87.63"`` pair, a city in the built-in table,
        or anything the U.S. Census geocoder understands.
        """
        if not isinstance(location, str) or not location.strip():
            raise LocationResolutionError("Location must be a non-empty string.")

        raw = location.strip()

        match = _LAT_LON_RE.match(raw)
        if match:
            lat, lon = float(match.group(1)), float(match.group(2))
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise LocationResolutionError(
                    f"Coordinates out of range: {raw!r}"
                )
            return lat, lon

        key = re.sub(r"\s+", " ", raw.lower())
        if key in US_CITY_COORDINATES:
            return US_CITY_COORDINATES[key]

        coords = self._geocode_with_census(raw)
        if coords is None:
            raise LocationResolutionError(
                f"Could not resolve {raw!r}. Use 'City, ST' for a U.S. city or "
                "a 'latitude,longitude' pair."
            )
        return coords

    def _geocode_with_census(self, address: str) -> tuple[float, float] | None:
        """Best-effort lookup against the free U.S. Census geocoder."""
        try:
            resp = self._session.get(
                _CENSUS_GEOCODER_URL,
                params={
                    "address": address,
                    "benchmark": "Public_AR_Current",
                    "format": "json",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            matches = (
                resp.json().get("result", {}).get("addressMatches", []) or []
            )
        except (requests.RequestException, ValueError):
            return None

        if not matches:
            return None
        coordinates = matches[0].get("coordinates") or {}
        lat, lon = coordinates.get("y"), coordinates.get("x")
        if lat is None or lon is None:
            return None
        return float(lat), float(lon)

    def resolve_grid_point(self, latitude: float, longitude: float) -> dict:
        """Resolve coordinates to an NWS forecast office and grid cell.

        ``GET /points/{lat},{lon}`` is the entry point for every gridded
        product. It also hands back the nearest named place, which is what we
        store as the canonical ``location`` label.
        """
        data = self.get(f"/points/{latitude:.4f},{longitude:.4f}")
        props = data.get("properties", {}) or {}
        relative = (props.get("relativeLocation") or {}).get("properties") or {}

        city, state = relative.get("city"), relative.get("state")
        label = f"{city}, {state}" if city and state else f"{latitude:.4f},{longitude:.4f}"

        return {
            "grid_office": props.get("gridId"),
            "grid_x": props.get("gridX"),
            "grid_y": props.get("gridY"),
            "forecast_url": props.get("forecast"),
            "forecast_hourly_url": props.get("forecastHourly"),
            "time_zone": props.get("timeZone"),
            "resolved_label": label,
            "latitude": latitude,
            "longitude": longitude,
        }

    # -- Raw product fetches ---------------------------------------------

    def get_active_alerts(self, latitude: float, longitude: float) -> list[dict]:
        """Active watches, warnings and advisories covering a point."""
        data = self.get(
            "/alerts/active", params={"point": f"{latitude:.4f},{longitude:.4f}"}
        )
        return data.get("features", []) or []

    def get_forecast(self, grid: dict) -> list[dict]:
        """The multi-day narrative forecast for a grid cell."""
        office, x, y = grid["grid_office"], grid["grid_x"], grid["grid_y"]
        if not office or x is None or y is None:
            return []
        data = self.get(f"/gridpoints/{office}/{x},{y}/forecast")
        return (data.get("properties", {}) or {}).get("periods", []) or []

    # -- Normalization ----------------------------------------------------

    def normalize_alert(self, feature: dict, grid: dict) -> dict | None:
        """Turn one alert GeoJSON feature into a weather document."""
        props = feature.get("properties", {}) or {}
        alert_id = props.get("id") or feature.get("id")
        if not alert_id:
            return None

        # The description carries the meteorological narrative and the
        # instruction carries the protective action ("Turn around, don't
        # drown"). Retrieval is far more useful with both, so they are
        # embedded as one document rather than two.
        parts = [
            (props.get("headline") or "").strip(),
            (props.get("description") or "").strip(),
            (props.get("instruction") or "").strip(),
        ]
        narrative = "\n\n".join(part for part in parts if part)
        if not narrative:
            return None

        return {
            "id": f"alert:{alert_id}",
            "location": grid["resolved_label"],
            "latitude": grid["latitude"],
            "longitude": grid["longitude"],
            "grid_office": grid["grid_office"],
            "grid_x": grid["grid_x"],
            "grid_y": grid["grid_y"],
            "source_type": SOURCE_TYPE_ALERT,
            "event": props.get("event"),
            "headline": props.get("headline") or props.get("event"),
            "severity": props.get("severity"),
            "urgency": props.get("urgency"),
            "certainty": props.get("certainty"),
            "area_desc": props.get("areaDesc"),
            "narrative_text": narrative,
            "issued_at": _parse_timestamp(props.get("sent")),
            "effective_at": _parse_timestamp(
                props.get("onset") or props.get("effective")
            ),
            "expires_at": _parse_timestamp(props.get("ends") or props.get("expires")),
            "content_hash": content_hash(narrative),
            "payload": feature,
        }

    def normalize_forecast_period(self, period: dict, grid: dict) -> dict | None:
        """Turn one forecast period into a weather document."""
        narrative = (period.get("detailedForecast") or "").strip()
        if not narrative:
            return None

        start = _parse_timestamp(period.get("startTime"))
        name = period.get("name") or f"period-{period.get('number', 0)}"

        # Dedup key: the same grid cell + period start always refers to the
        # same slice of time, so a re-issued forecast updates the row in place
        # instead of piling up near-duplicates.
        doc_id = (
            f"forecast:{grid['grid_office']}:{grid['grid_x']},{grid['grid_y']}:"
            f"{start or 'unknown'}:{_slugify(name)}"
        )

        short = period.get("shortForecast") or name
        headline = f"{name}: {short}"

        return {
            "id": doc_id,
            "location": grid["resolved_label"],
            "latitude": grid["latitude"],
            "longitude": grid["longitude"],
            "grid_office": grid["grid_office"],
            "grid_x": grid["grid_x"],
            "grid_y": grid["grid_y"],
            "source_type": SOURCE_TYPE_FORECAST,
            "event": short,
            "headline": headline,
            "severity": None,
            "urgency": None,
            "certainty": None,
            "area_desc": grid["resolved_label"],
            "narrative_text": narrative,
            "issued_at": start,
            "effective_at": start,
            "expires_at": _parse_timestamp(period.get("endTime")),
            "content_hash": content_hash(narrative),
            "payload": period,
        }

    # -- Public entry point ----------------------------------------------

    def fetch_documents(
        self,
        locations: Iterable[str],
        limit: int = 50,
        include_alerts: bool = True,
        include_forecast: bool = True,
    ) -> tuple[list[dict], list[dict]]:
        """Harvest normalized documents for a list of locations.

        Returns ``(documents, errors)``. A location that fails to resolve or
        whose products are unavailable is reported in ``errors`` and skipped,
        so one bad entry never sinks the whole sync.
        """
        documents: list[dict] = []
        errors: list[dict] = []
        seen_ids: set[str] = set()

        for location in locations:
            try:
                latitude, longitude = self.resolve_location(location)
                grid = self.resolve_grid_point(latitude, longitude)
            except (LocationResolutionError, requests.RequestException) as exc:
                errors.append({"location": location, "error": str(exc)})
                continue

            per_location: list[dict] = []

            if include_alerts:
                try:
                    for feature in self.get_active_alerts(latitude, longitude):
                        doc = self.normalize_alert(feature, grid)
                        if doc:
                            per_location.append(doc)
                except requests.RequestException as exc:
                    errors.append(
                        {"location": location, "error": f"alerts unavailable: {exc}"}
                    )

            if include_forecast:
                try:
                    for period in self.get_forecast(grid):
                        doc = self.normalize_forecast_period(period, grid)
                        if doc:
                            per_location.append(doc)
                except requests.RequestException as exc:
                    errors.append(
                        {"location": location, "error": f"forecast unavailable: {exc}"}
                    )

            # Alerts come first so that when `limit` bites, the safety-critical
            # documents are the ones that survive.
            per_location.sort(key=lambda d: d["source_type"] != SOURCE_TYPE_ALERT)

            for doc in per_location[:limit]:
                if doc["id"] in seen_ids:
                    continue
                seen_ids.add(doc["id"])
                documents.append(doc)

        return documents, errors

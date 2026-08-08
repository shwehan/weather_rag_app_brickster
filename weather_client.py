"""National Weather Service client and weather-document normalizer."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any

import requests


NWS_BASE_URL = os.environ.get("NWS_BASE_URL", "https://api.weather.gov")
GEOCODER_BASE_URL = os.environ.get(
    "GEOCODER_BASE_URL", "https://geocoding-api.open-meteo.com/v1"
)
DEFAULT_TIMEOUT = 30
COORDINATE_RE = re.compile(
    r"^\s*(-?(?:\d+(?:\.\d+)?|\.\d+))\s*,\s*"
    r"(-?(?:\d+(?:\.\d+)?|\.\d+))\s*$"
)


class WeatherClientError(RuntimeError):
    """Raised when a location or upstream weather request cannot be resolved."""


@dataclass(frozen=True)
class Location:
    label: str
    latitude: float
    longitude: float


class WeatherClient:
    def __init__(
        self,
        nws_base_url: str | None = None,
        geocoder_base_url: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ):
        self.nws_base_url = (nws_base_url or NWS_BASE_URL).rstrip("/")
        self.geocoder_base_url = (geocoder_base_url or GEOCODER_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": os.environ.get(
                    "NWS_USER_AGENT",
                    "weather-intelligence/1.0 (github.com/shwehan)",
                ),
                "Accept": "application/geo+json, application/json",
            }
        )

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict:
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise WeatherClientError(f"Upstream request failed: {exc}") from exc
        except ValueError as exc:
            raise WeatherClientError("Upstream service returned invalid JSON") from exc

    def resolve_location(self, value: str | dict[str, Any]) -> Location:
        """Resolve `lat,lon`, a mapping, or a city/state string to coordinates."""
        if isinstance(value, dict):
            try:
                latitude = float(value["latitude"])
                longitude = float(value["longitude"])
            except (KeyError, TypeError, ValueError) as exc:
                raise WeatherClientError(
                    "Location objects require numeric latitude and longitude"
                ) from exc
            label = str(value.get("label") or f"{latitude:.4f},{longitude:.4f}").strip()
            return self._validated_location(label, latitude, longitude)

        if not isinstance(value, str) or not value.strip():
            raise WeatherClientError("Locations must be non-empty strings or objects")

        raw = value.strip()
        match = COORDINATE_RE.match(raw)
        if match:
            return self._validated_location(raw, float(match.group(1)), float(match.group(2)))

        # NWS needs coordinates. Open-Meteo is used only for place-name geocoding;
        # all weather text still comes from api.weather.gov.
        data = self._get(
            f"{self.geocoder_base_url}/search",
            params={
                "name": raw,
                "count": 10,
                "language": "en",
                "format": "json",
                "countryCode": "US",
            },
        )
        results = data.get("results") or []
        if not results:
            raise WeatherClientError(f"Could not resolve U.S. location: {raw}")

        best = self._best_geocoder_match(raw, results)
        label_parts = [best.get("name"), best.get("admin1")]
        label = ", ".join(str(part) for part in label_parts if part) or raw
        return self._validated_location(
            label, float(best["latitude"]), float(best["longitude"])
        )

    @staticmethod
    def _best_geocoder_match(query: str, results: list[dict]) -> dict:
        """Prefer an exact state match for inputs such as `Chicago, IL`."""
        parts = [part.strip().lower() for part in query.split(",")]
        if len(parts) < 2:
            return results[0]
        state_query = parts[-1]
        for result in results:
            admin = str(result.get("admin1") or "").lower()
            admin_code = str(result.get("admin1_code") or "").lower()
            if state_query in {admin, admin_code}:
                return result
        return results[0]

    @staticmethod
    def _validated_location(label: str, latitude: float, longitude: float) -> Location:
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise WeatherClientError("Latitude or longitude is out of range")
        return Location(label=label[:200], latitude=latitude, longitude=longitude)

    def fetch_documents(self, value: str | dict[str, Any], limit: int = 50) -> list[dict]:
        """Fetch active alerts and multi-day forecast narratives for one location."""
        location = self.resolve_location(value)
        point = f"{location.latitude:.4f},{location.longitude:.4f}"
        point_data = self._get(f"{self.nws_base_url}/points/{point}")
        point_properties = point_data.get("properties") or {}
        forecast_url = point_properties.get("forecast")
        if not forecast_url:
            raise WeatherClientError(f"NWS returned no forecast URL for {location.label}")

        alerts_data = self._get(
            f"{self.nws_base_url}/alerts/active", params={"point": point}
        )
        forecast_data = self._get(forecast_url)

        documents = self._normalize_alerts(location, alerts_data)
        documents.extend(self._normalize_forecast(location, forecast_url, forecast_data))
        return documents[: max(1, min(int(limit), 200))]

    def _normalize_alerts(self, location: Location, data: dict) -> list[dict]:
        documents = []
        for feature in data.get("features") or []:
            props = feature.get("properties") or {}
            description = (props.get("description") or "").strip()
            instruction = (props.get("instruction") or "").strip()
            narrative = "\n\n".join(part for part in (description, instruction) if part)
            if not narrative:
                continue
            upstream_id = str(feature.get("id") or props.get("id") or "")
            stable_id = self._stable_id(
                "alert", location.label, upstream_id or hashlib.sha256(narrative.encode()).hexdigest()
            )
            documents.append(
                self._document(
                    stable_id=stable_id,
                    location=location,
                    source_type="alert",
                    headline=props.get("headline") or props.get("event") or "Weather alert",
                    narrative=narrative,
                    issued_at=props.get("sent") or props.get("onset"),
                    effective_at=props.get("effective") or props.get("onset"),
                    payload=feature,
                )
            )
        return documents

    def _normalize_forecast(
        self, location: Location, forecast_url: str, data: dict
    ) -> list[dict]:
        documents = []
        periods = (data.get("properties") or {}).get("periods") or []
        for period in periods:
            narrative = (period.get("detailedForecast") or "").strip()
            if not narrative:
                continue
            issued_at = (data.get("properties") or {}).get("generatedAt")
            effective_at = period.get("startTime")
            upstream_key = "|".join(
                [forecast_url, str(period.get("number")), str(effective_at)]
            )
            headline = " — ".join(
                part
                for part in (period.get("name"), period.get("shortForecast"))
                if part
            ) or "Weather forecast"
            documents.append(
                self._document(
                    stable_id=self._stable_id("forecast", location.label, upstream_key),
                    location=location,
                    source_type="forecast",
                    headline=headline,
                    narrative=narrative,
                    issued_at=issued_at,
                    effective_at=effective_at,
                    payload=period,
                )
            )
        return documents

    @staticmethod
    def _stable_id(*parts: str) -> str:
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
        return f"weather:{digest}"

    @staticmethod
    def _document(
        *,
        stable_id: str,
        location: Location,
        source_type: str,
        headline: str,
        narrative: str,
        issued_at: str | None,
        effective_at: str | None,
        payload: dict,
    ) -> dict:
        content_hash = hashlib.sha256(narrative.encode("utf-8")).hexdigest()
        return {
            "id": stable_id,
            "location": location.label,
            "latitude": location.latitude,
            "longitude": location.longitude,
            "source_type": source_type,
            "headline": str(headline)[:500],
            "narrative_text": narrative,
            "issued_at": issued_at,
            "effective_at": effective_at,
            "payload": payload,
            "content_hash": content_hash,
        }

import unittest

from embedding_model import chunk_text, vector_literal
from weather_client import Location, WeatherClient, WeatherClientError


class ChunkingTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(chunk_text("  sunny   and warm  "), ["sunny and warm"])

    def test_long_text_overlaps(self):
        chunks = chunk_text("one two three four five six seven", chunk_size=18, overlap=4)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks))

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("weather", chunk_size=5, overlap=5)

    def test_vector_literal(self):
        self.assertEqual(vector_literal([0.1, -0.2]), "[0.1,-0.2]")


class WeatherClientTests(unittest.TestCase):
    def setUp(self):
        self.client = WeatherClient()

    def test_coordinate_string(self):
        resolved = self.client.resolve_location("41.8781,-87.6298")
        self.assertEqual(resolved, Location("41.8781,-87.6298", 41.8781, -87.6298))

    def test_location_object(self):
        resolved = self.client.resolve_location(
            {"label": "Austin, TX", "latitude": 30.2672, "longitude": -97.7431}
        )
        self.assertEqual(resolved.label, "Austin, TX")

    def test_out_of_range_coordinate(self):
        with self.assertRaises(WeatherClientError):
            self.client.resolve_location("99,-200")

    def test_alert_normalization_combines_instruction(self):
        location = Location("Test, TX", 30.0, -97.0)
        docs = self.client._normalize_alerts(
            location,
            {
                "features": [
                    {
                        "id": "alert-1",
                        "properties": {
                            "event": "Flood Warning",
                            "description": "Water is rising.",
                            "instruction": "Move to higher ground.",
                            "sent": "2026-08-01T00:00:00Z",
                        },
                    }
                ]
            },
        )
        self.assertEqual(len(docs), 1)
        self.assertIn("Move to higher ground", docs[0]["narrative_text"])
        self.assertEqual(docs[0]["source_type"], "alert")

    def test_forecast_id_stays_stable_when_text_changes(self):
        location = Location("Test, TX", 30.0, -97.0)
        base_period = {
            "number": 1,
            "name": "Tonight",
            "shortForecast": "Rain",
            "startTime": "2026-08-07T18:00:00-05:00",
        }
        first = self.client._normalize_forecast(
            location,
            "https://api.weather.gov/gridpoints/EWX/1,1/forecast",
            {
                "properties": {
                    "generatedAt": "2026-08-07T12:00:00Z",
                    "periods": [{**base_period, "detailedForecast": "Light rain."}],
                }
            },
        )[0]
        changed = self.client._normalize_forecast(
            location,
            "https://api.weather.gov/gridpoints/EWX/1,1/forecast",
            {
                "properties": {
                    "generatedAt": "2026-08-07T13:00:00Z",
                    "periods": [{**base_period, "detailedForecast": "Heavy rain."}],
                }
            },
        )[0]
        self.assertEqual(first["id"], changed["id"])
        self.assertNotEqual(first["content_hash"], changed["content_hash"])


if __name__ == "__main__":
    unittest.main()

from qhana_plugin_runner.requests import get_mimetype

from tests.utils import MockResponse


def test_get_mimetype_strips_content_type_parameters():
    response = MockResponse("http://example.com/vector.csv", "text/csv; charset=utf-8")

    assert get_mimetype(response) == "text/csv"


def test_get_mimetype_falls_back_to_url_extension_when_header_missing():
    response = MockResponse("http://example.com/vector.json", "", content=b"{}")
    response.headers.pop("Content-Type", None)

    assert get_mimetype(response) == "application/json"

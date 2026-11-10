import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "helpers",
    Path(__file__).resolve().parents[1] / "src" / "utils" / "helpers.py",
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def test_clean_text_strips_urls_without_eating_http_words() -> None:
    assert helpers.clean_text("see https://example.com now") == "see now"
    assert helpers.clean_text("see http://example.com/a now") == "see now"
    assert helpers.clean_text("visit www.example.com please") == "visit please"
    # The old www. pattern treated the unescaped dot as any char, so
    # "wwwXexample" was stripped. The new pattern requires a literal dot.
    assert helpers.clean_text("the wwwXexample token stays") == "the wwwXexample token stays"
    # The old http\S+ pattern ate any "http" prefix, including "httpstatus".
    assert helpers.clean_text("the httpstatus code is 200") == "the httpstatus code is 200"

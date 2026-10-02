"""Tests for Hindi/Hinglish detection, normalization and ONNX translation.

The pure-logic tests (detection, normalization) are fast and never touch the
model. The translation tests load the real fp16 ONNX graphs from
``app/translation_assets/`` once per module, mirroring
``test_sentiment_service.py``.
"""

from __future__ import annotations

import pytest

from app import config
from app.services.translation import (
    HindiTranslator,
    contains_devanagari,
    detect_language,
    prepare_text,
    roman_to_devanagari,
)


# --------------------------------------------------------------------------
# script detection
# --------------------------------------------------------------------------


def test_contains_devanagari() -> None:
    assert contains_devanagari("मुझे यह पसंद है")
    assert not contains_devanagari("I like this")
    assert not contains_devanagari("")


# --------------------------------------------------------------------------
# language detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I absolutely love this product!",
        "This is the worst purchase I have ever made.",
        "The package arrived three days late and the box was crushed.",
        "It works exactly as described, nothing more and nothing less.",
        "I am not impressed with the quality at all.",
        "Delivery was quick and the item works great for the price.",
        "It is okay, nothing special.",
        "Please contact support at help@example.com",
        "I love apples and oranges",
        "movie was great but the 3D quality is not good",
        "The food was amazing but the waiter was rude",
    ],
)
def test_english_is_never_flagged_as_hindi(text: str) -> None:
    """The most important property: English must pass through untouched."""
    assert detect_language(text) == "en"


@pytest.mark.parametrize(
    "text",
    [
        "मुझे यह उत्पाद बहुत अच्छा लगा।",
        "यह उत्पाद बहुत खराब है।",
        "आज मौसम बहुत ठंडा है।",
    ],
)
def test_devanagari_detected_as_hindi(text: str) -> None:
    assert detect_language(text) == "hi"


@pytest.mark.parametrize(
    "text",
    [
        "Mujhe ye product bahut achha laga",
        "Mujhe ye product bilkul pasand nahi hai",
        "Ye product bahut kharab hai",
        "Ye movie bahut achhi thi",
        "aaj mausam bahut kharab hai",
        "sab kuch perfect tha, staff bahut helpful tha",
        "kripya jaldi reply kijiye",
        "mera dost bimar hai",
    ],
)
def test_roman_hindi_detected_as_hinglish(text: str) -> None:
    assert detect_language(text) == "hinglish"


@pytest.mark.parametrize("text", ["", "   ", "12345", "!!!"])
def test_empty_or_neutral_input_is_english(text: str) -> None:
    assert detect_language(text) == "en"


# --------------------------------------------------------------------------
# regression: proper nouns and generic words must not trigger translation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "My friend Mukesh told me the name of the restaurant",
        "The name of the restaurant is printed on the receipt",
        "My name is Andrew and I work at Google",
        "Vijay said the quality was not good at all",
        "Please contact the support team about your order",
    ],
)
def test_english_with_names_and_generic_words_is_not_hinglish(text: str) -> None:
    """Regression: a person named "Mukesh" and the word "name" are not Hindi.

    Both used to be counted as weak Hinglish markers, so this ordinary English
    sentence was routed through the MT model. Names carry no language
    information, and "name" is a common English word, so neither may promote a
    sentence to Hinglish.
    """
    assert detect_language(text) == "en"


def test_prepare_text_leaves_english_with_a_name_untouched() -> None:
    """The full English path must be used: no model load, no translation."""
    text = "My friend Mukesh told me the name of the restaurant"
    result = prepare_text(text)
    assert result.language == "en"
    assert result.status == "not_needed"
    assert result.text_for_model == text
    assert result.translated_text is None


# --------------------------------------------------------------------------
# Roman -> Devanagari normalization
# --------------------------------------------------------------------------


def test_normalization_maps_function_words() -> None:
    result = roman_to_devanagari("mujhe ye product achha hai")
    assert "मुझे" in result
    assert "यह" in result
    assert "उत्पाद" in result
    assert "अच्छा" in result
    assert "है" in result


def test_normalization_keeps_unknown_tokens_untouched() -> None:
    """Unmapped tokens stay in Latin rather than being transliterated.

    Blind transliteration of English words produced garbage that the MT model
    then mistranslated ("delivery" -> "taber"), so unknown words are the
    deliberate, gracefully-degrading fallback.
    """
    result = roman_to_devanagari("blah woh kharab xyzzy")
    assert "blah" in result
    assert "xyzzy" in result
    assert "खराब" in result


def test_normalization_preserves_punctuation_and_case() -> None:
    assert roman_to_devanagari("Kharab hai, bilkul!") == "खराब है, बिल्कुल!"


def test_normalization_is_idempotent_on_devanagari() -> None:
    text = "मुझे यह उत्पाद बहुत अच्छा लगा।"
    assert roman_to_devanagari(text) == text


def test_normalization_handles_empty_text() -> None:
    assert roman_to_devanagari("") == ""


# --------------------------------------------------------------------------
# real ONNX translation
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def translator() -> HindiTranslator:
    return HindiTranslator()


def test_translator_loads_real_assets(translator: HindiTranslator) -> None:
    assert translator.load() is translator
    assert translator.loaded is True


@pytest.mark.parametrize(
    ("hindi", "expected_fragments"),
    [
        ("मुझे यह उत्पाद बहुत अच्छा लगा।", ("love", "product")),
        ("मुझे यह उत्पाद बिल्कुल पसंद नहीं है।", ("don't like", "product")),
        ("यह उत्पाद बहुत खराब है।", ("very bad",)),
        ("यह मूवी बहुत अच्छी थी।", ("movie", "good")),
    ],
)
def test_devanagari_translation_quality(
    translator: HindiTranslator,
    hindi: str,
    expected_fragments: tuple[str, ...],
) -> None:
    english = translator.translate(hindi).lower()
    assert english
    for fragment in expected_fragments:
        assert fragment in english


def test_translation_rejects_bad_input(translator: HindiTranslator) -> None:
    with pytest.raises(TypeError):
        translator.translate(42)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        translator.translate("   ")


def test_translation_truncates_long_input(translator: HindiTranslator) -> None:
    """A long input must not raise; it is truncated to the encoder limit."""
    long_text = "यह उत्पाद बहुत अच्छा है। " * 400
    assert translator.translate(long_text)


# --------------------------------------------------------------------------
# prepare_text orchestration
# --------------------------------------------------------------------------


def test_prepare_text_skips_english_without_loading_the_model() -> None:
    result = prepare_text("I absolutely love this product!")
    assert result.language == "en"
    assert result.status == "not_needed"
    assert result.text_for_model == "I absolutely love this product!"
    assert result.translated_text is None


def test_prepare_text_translates_hinglish() -> None:
    result = prepare_text("Mujhe ye product bahut achha laga")
    assert result.language == "hinglish"
    assert result.status == "translated"
    assert result.original_text == "Mujhe ye product bahut achha laga"
    assert result.translated_text
    assert "product" in result.translated_text.lower()
    # the sentiment model reads the English, not the Romanized Hindi
    assert result.text_for_model == result.translated_text
    assert result.text_for_model.isascii()


def test_prepare_text_translates_devanagari() -> None:
    result = prepare_text("यह उत्पाद बहुत खराब है।")
    assert result.language == "hi"
    assert result.status == "translated"
    assert "bad" in result.translated_text.lower()


def test_prepare_text_survives_translation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken translator degrades to the original text, never an exception."""
    import app.services.translation as mod

    class Broken:
        loaded = False

        def translate(self, text: str) -> str:
            raise RuntimeError("boom")

    monkeypatch.setattr(mod, "get_translator", lambda: Broken())
    result = prepare_text("Ye product bahut kharab hai")
    assert result.status == "failed"
    assert result.language == "hinglish"
    assert result.translated_text is None
    assert result.text_for_model == "Ye product bahut kharab hai"


def test_prepare_text_survives_empty_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.translation as mod

    class Empty:
        loaded = False

        def translate(self, text: str) -> str:
            return "   "

    monkeypatch.setattr(mod, "get_translator", lambda: Empty())
    result = prepare_text("Ye product bahut kharab hai")
    assert result.status == "failed"
    assert result.text_for_model == "Ye product bahut kharab hai"


def test_translation_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "TRANSLATION_ENABLED", False)
    result = prepare_text("Ye product bahut kharab hai")
    assert result.status == "not_needed"
    assert result.text_for_model == "Ye product bahut kharab hai"
    assert result.translated_text is None


def test_get_translator_returns_singleton() -> None:
    from app.services.translation import get_translator

    assert get_translator() is get_translator()
    assert isinstance(get_translator(), HindiTranslator)

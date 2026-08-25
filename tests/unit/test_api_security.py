"""Unit tests for the API's request-level protections (master spec §40).

These are the pieces small enough to test directly, without a client: filename
sanitising, the API-key comparison, and the header set. The integration suite
covers how they behave over HTTP; this covers whether they are individually
correct, including the cases an HTTP test cannot easily reach — a key configured
but not required, a name that is nothing but punctuation, a truncation that would
otherwise eat a file extension.

:func:`~src.api.security.safe_filename` has no caller yet. It is tested anyway,
because the PDF milestone will reach for it under time pressure and the failure
mode of a filename sanitiser is silent: a name that looks clean but still carries
a separator, an override character, or a lost extension.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.api.security import SECURITY_HEADERS, check_api_key, safe_filename
from src.config.settings import Settings

#: Names that must not survive intact. Each pair is ``(input, what must be true)``
#: rather than an exact expected output, because the point is the property — no
#: separator, no traversal, no control character — not one particular spelling.
HOSTILE_NAMES = [
    "../../etc/passwd",
    "..\\..\\windows\\system32\\config\\sam",
    "/etc/shadow",
    "C:\\Users\\someone\\.env",
    "....//....//secret.pdf",
    "paper\x00.pdf",
    "pap\ner.pdf",
    "\u202egnp.exe",  # right-to-left override: displays as "exe.png"
    "résumé.pdf",
    "  spaced name .pdf",
    "..",
    ".",
    "...",
    "___",
    "",
]


# ---------------------------------------------------------------------------
# safe_filename
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", HOSTILE_NAMES)
def test_a_sanitised_name_is_always_a_plain_basename(raw: str) -> None:
    """Whatever arrives, what comes back is a single safe segment.

    Asserted as invariants rather than exact strings: a future change to the
    allowlist should be free to spell the result differently, but must not be free
    to emit a separator, a leading dot, or an empty name.
    """
    cleaned = safe_filename(raw)

    assert cleaned, "a sanitised name is never empty"
    assert "/" not in cleaned and "\\" not in cleaned, "no path separator survives"
    assert ".." not in cleaned, "no traversal sequence survives"
    assert not cleaned.startswith("."), "no hidden-file name is produced"
    assert cleaned == cleaned.strip(), "no leading or trailing whitespace"
    assert all(char.isalnum() or char in "._-" for char in cleaned), f"unsafe char in {cleaned!r}"
    assert len(cleaned) <= 96


def test_traversal_is_reduced_to_the_final_segment() -> None:
    """Both separator conventions are stripped, not just the platform's own.

    A name arriving from a Windows client may use backslashes, which POSIX
    ``PurePath`` treats as ordinary characters — so a sanitiser that trusts the
    platform's own path parsing lets ``..\\..\\x`` through unchanged on Linux.
    """
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("..\\..\\secret.pdf") == "secret.pdf"
    assert safe_filename("a/b\\c/d.pdf") == "d.pdf"


def test_a_name_with_nothing_usable_falls_back() -> None:
    """Punctuation-only and empty names get the fallback stem.

    ``Path("..").name`` is ``".."`` — the case where the obvious implementation
    returns something actively dangerous rather than merely useless.
    """
    for raw in ("..", ".", "...", "///", "___", "", "   "):
        assert safe_filename(raw) == "upload", f"{raw!r} did not fall back"

    assert safe_filename("!!!", fallback="paper") == "paper"


def test_format_and_control_characters_are_dropped() -> None:
    """Bidi overrides and NULs are removed before the allowlist runs.

    Order matters. A right-to-left override makes ``evil\u202egnp.exe`` display as
    ``evilexe.png``, so a name that passes an extension check by eye can still be
    an executable. Stripping the class of character is the fix; escaping it is not.
    """
    assert "\u202e" not in safe_filename("evil\u202egnp.exe")
    assert "\x00" not in safe_filename("paper\x00.pdf")
    assert safe_filename("paper\x00.pdf") == "paper.pdf"


def test_unicode_is_normalised_before_filtering() -> None:
    """NFKC first, so a decomposed or compatibility form cannot smuggle anything.

    Normalising after the character filter would let a compatibility character
    expand into a separator once something downstream normalised it.
    """
    # U+FF0F FULLWIDTH SOLIDUS normalises to "/", which the separator split then
    # removes — the whole reason normalisation runs first.
    assert safe_filename("a\uff0fb.pdf") == "b.pdf"
    # A decomposed "é" normalises to a single codepoint, then falls outside the
    # ASCII allowlist and is collapsed rather than silently truncating the name.
    assert safe_filename("re\u0301sume\u0301.pdf").endswith(".pdf")


def test_truncation_keeps_the_extension() -> None:
    """A length cap must not turn a validated ``.pdf`` into an extensionless file.

    The extension is what a downstream format check reads. Cutting it off would
    make a long-named but valid upload fail validation for the wrong reason.
    """
    cleaned = safe_filename("x" * 300 + ".pdf")

    assert cleaned.endswith(".pdf")
    assert len(cleaned) <= 96

    # A "suffix" too long to be a real extension is not preserved at the cost of
    # the stem — it is just a long name, truncated.
    odd = safe_filename("y" * 200 + "." + "z" * 40)
    assert len(odd) <= 96


def test_an_already_safe_name_is_left_alone() -> None:
    """Sanitising is not mangling: a clean name round-trips unchanged."""
    for name in ("paper.pdf", "2024_smith_et-al.pdf", "S00001.json"):
        assert safe_filename(name) == name


@pytest.mark.parametrize(
    "raw", ["CON", "con.pdf", "NUL.txt", "aux", "COM1.pdf", "lpt9.json", "PRN.pdf"]
)
def test_windows_device_names_are_defused(raw: str) -> None:
    """A reserved device name is made ordinary while staying recognisable.

    This is the one hazard the character allowlist cannot see, because every
    character in ``CON.pdf`` is permitted. Windows resolves the name as a device
    regardless of extension or directory, so opening it for writing writes to the
    console rather than to a file — on the platform this project is developed on.
    """
    cleaned = safe_filename(raw)
    stem = cleaned.partition(".")[0]

    assert stem.upper() not in {"CON", "PRN", "AUX", "NUL", "COM1", "LPT9"}
    assert cleaned.lower().startswith(raw.partition(".")[0].lower()), "still recognisable"
    if "." in raw:
        assert cleaned.endswith(raw[raw.index(".") :]), "the extension is preserved"


# ---------------------------------------------------------------------------
# check_api_key
# ---------------------------------------------------------------------------
def configured(settings: Settings, *, require: bool, key: str | None) -> Settings:
    """Return settings with the API-key flag and secret set."""
    security = settings.api.security.model_copy(update={"require_api_key": require})
    return settings.model_copy(
        update={
            "api": settings.api.model_copy(update={"security": security}),
            "env": settings.env.model_copy(update={"aris_api_key": key}),
        }
    )


@pytest.mark.parametrize("supplied", [None, "", "anything", "wrong"])
def test_no_key_is_needed_when_enforcement_is_off(settings: Settings, supplied: str | None) -> None:
    """With the flag off, nothing is checked — including a configured key.

    A key present in the environment with the flag off is a key staged for later,
    not an accidental half-enabled state. Enforcing on the key's mere presence
    would lock out a developer who exported it while preparing a deployment.
    """
    check_api_key(configured(settings, require=False, key="staged-secret"), supplied)


def test_the_matching_key_is_accepted(settings: Settings) -> None:
    """The happy path raises nothing."""
    check_api_key(configured(settings, require=True, key="right"), "right")


@pytest.mark.parametrize("supplied", [None, "", "wrong", "righ", "rights", "RIGHT", " right"])
def test_a_missing_or_wrong_key_is_a_401(settings: Settings, supplied: str | None) -> None:
    """Absent, empty, near-miss, and wrong-case keys all fail closed.

    Prefixes and suffixes are included deliberately: the comparison is
    ``secrets.compare_digest``, and a short-circuiting ``==`` would leak the length
    of the matching prefix through response timing.
    """
    with pytest.raises(HTTPException) as raised:
        check_api_key(configured(settings, require=True, key="right"), supplied)

    assert raised.value.status_code == 401
    assert raised.value.headers == {"WWW-Authenticate": "X-API-Key"}
    assert "right" not in raised.value.detail, "the expected key is never disclosed"


def test_enforcement_with_no_configured_key_is_a_500_naming_the_variable(
    settings: Settings,
) -> None:
    """Required-but-unset is a misconfiguration, not an authentication failure.

    A 401 here would be unsatisfiable by any value and indistinguishable from a
    wrong key, sending the operator to debug the client. The 500 names the
    variable and the file to set it in.
    """
    for key in (None, ""):
        with pytest.raises(HTTPException) as raised:
            check_api_key(configured(settings, require=True, key=key), "anything")

        assert raised.value.status_code == 500
        assert "ARIS_API_KEY" in raised.value.detail
        assert ".env" in raised.value.detail


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------
def test_scripts_get_no_inline_exemption() -> None:
    """``script-src`` is ``'self'`` with no ``'unsafe-inline'``.

    This is the half of the CSP that matters for XSS. Styles do carry the
    exemption — the dashboard sets computed bar widths and chart geometry through
    the ``style`` attribute, which no nonce can cover — and keeping the two
    separated is the point: the loose directive must not be copied across.
    """
    directives = {
        part.strip().split(" ")[0]: part.strip()
        for part in SECURITY_HEADERS["Content-Security-Policy"].split(";")
        if part.strip()
    }

    assert directives["script-src"] == "script-src 'self'"
    assert "unsafe-inline" not in directives["script-src"]
    assert "unsafe-eval" not in SECURITY_HEADERS["Content-Security-Policy"]
    assert "unsafe-inline" in directives["style-src"], "documented, and styles only"


def test_the_policy_closes_the_directives_a_dashboard_does_not_need() -> None:
    """Plugins, framing, form posts, and base-tag rewriting are all denied.

    The dashboard loads its own CSS and ES modules, renders inline SVG, and posts
    JSON with ``fetch``. Everything outside that is switched off rather than left
    to the ``default-src`` fallback, which does not cover ``base-uri`` or
    ``form-action`` at all.
    """
    policy = SECURITY_HEADERS["Content-Security-Policy"]

    for directive in (
        "default-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
        "connect-src 'self'",
        "img-src 'self' data:",
    ):
        assert directive in policy, f"missing {directive}"


def test_the_header_set_covers_sniffing_framing_and_referrers() -> None:
    """The non-CSP headers are present and set to their strict values."""
    assert SECURITY_HEADERS["X-Content-Type-Options"] == "nosniff"
    assert SECURITY_HEADERS["X-Frame-Options"] == "DENY"
    assert SECURITY_HEADERS["Referrer-Policy"] == "no-referrer"
    assert SECURITY_HEADERS["Cross-Origin-Opener-Policy"] == "same-origin"

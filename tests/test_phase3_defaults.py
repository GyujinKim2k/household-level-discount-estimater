"""The Phase 3 defaults, pinned where they can silently drift apart.

Two of these encode mistakes that were actually made, not hypothetical ones.

``start_age_high`` is *paired* with ``n_waves``: both arms of a wave comparison
must occupy the same age envelope, or window length is confounded with how deep
into life the window reaches. That confound flattered the 15-wave arm, was
flagged there, and then went unnoticed through the entire 5/6/7/8/9/10/15
comparison because the pairing lived only in a comment.

And the ensemble is not an optimization. A single flow undercovers reproducibly
(coverage ~0.87 against a nominal 0.900, seed sd <= 0.012); reporting one is
shipping a posterior whose stated uncertainty is known to be too tight. If
``n_members`` ever reads 1, that is a regression and not a config choice.
"""

from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "npe" / "phase3.yaml"
AGE_ENVELOPE_HIGH = 59  # SIMULATOR_SPEC 6.1.2


def _cfg():
    return yaml.safe_load(CONFIG.read_text())


def test_start_age_high_is_paired_with_n_waves():
    """high = 59 - wave_years * n_waves + 1, so the envelope ends at 59."""
    c = _cfg()
    n_waves = c["dataset"]["n_waves"]
    wave_years = c["dataset"]["wave_years"]
    expected = AGE_ENVELOPE_HIGH - wave_years * n_waves + 1
    assert c["window"]["start_age_high"] == expected, (
        f"n_waves={n_waves} needs start_age_high={expected}, config has "
        f"{c['window']['start_age_high']}. These move together; changing one "
        f"alone confounds window length with how far into life it reaches."
    )


def test_window_ends_before_retirement_pressure():
    """The envelope must stay short of the mortality-free regime (SPEC 4)."""
    c = _cfg()
    end = (c["window"]["start_age_high"]
           + c["dataset"]["wave_years"] * c["dataset"]["n_waves"] - 1)
    assert end == AGE_ENVELOPE_HIGH
    assert end < 64, "windows reach retirement; the forward pass has no mortality"


def test_ensemble_is_configured_and_plural():
    c = _cfg()
    assert "ensemble" in c, "Phase 3 requires an ensemble; see SPEC 6.1.3"
    n = c["ensemble"]["n_members"]
    assert n >= 2, (
        f"n_members={n}: a single flow undercovers (~0.87 vs nominal 0.900). "
        f"Ensembling is the fix, not a tuning knob."
    )


def test_n_waves_is_what_psid_can_supply():
    """7 biennial waves = PSID 2011-2023, the clean credit-card window."""
    c = _cfg()
    assert c["dataset"]["n_waves"] == 7
    assert c["dataset"]["wave_years"] == 2


def test_windows_per_panel_at_the_measured_plateau():
    c = _cfg()
    assert c["window"]["windows_per_panel"] == 8


def test_generator_default_matches_the_config():
    """A generator writing a different window than the config analyses is a trap."""
    import ast

    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "generate_dataset.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "add_argument"
                and node.args
                and getattr(node.args[0], "value", None) == "--n_waves"):
            default = next(ast.literal_eval(kw.value) for kw in node.keywords
                           if kw.arg == "default")
            assert default == _cfg()["dataset"]["n_waves"]
            return
    raise AssertionError("no --n_waves argument in generate_dataset.py")

from src.utils.kelly import kelly_fraction, kelly_position_usdc


def test_no_edge_returns_zero():
    assert kelly_fraction(0.30, 0.30) == 0
    assert kelly_fraction(0.20, 0.40) == 0


def test_positive_edge_positive_fraction():
    f = kelly_fraction(0.50, 0.30)
    assert f > 0


def test_position_capped_by_max_fraction():
    pos = kelly_position_usdc(0.95, 0.10, bankroll=1000.0,
                              fraction_multiplier=1.0, max_fraction=0.05)
    assert pos == 50.0  # 1000 * 0.05


def test_quarter_kelly_default():
    full = kelly_fraction(0.40, 0.20)
    pos = kelly_position_usdc(0.40, 0.20, bankroll=1000.0)
    expected = round(min(full * 0.25, 0.05) * 1000.0, 2)
    assert pos == expected

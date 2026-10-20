"""``oplot.my.vlines`` works on current matplotlib (i2mint/oplot#10: it passed a removed ``hold``)."""

import matplotlib

matplotlib.use("Agg")

from oplot.my import vlines


def test_vlines_draws_stems_for_a_bare_sequence():
    lines = vlines([1, 2, 3])
    assert len(lines.get_segments()) == 3

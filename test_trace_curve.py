import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import numpy as np
import pytest

from inpythotools.trace_curve import (
    _PointSetCollector,
    _convert_point_sets,
    _load_trace_image,
    _select_axis_corners,
    _validate_set_count,
)


def _mouse_event(ax, button, x=1.0, y=2.0):
    return SimpleNamespace(inaxes=ax, button=button, xdata=x, ydata=y)


def test_middle_click_starts_a_new_set():
    fig, ax = plt.subplots()
    collector = _PointSetCollector(ax)

    collector._on_button(_mouse_event(ax, MouseButton.LEFT))
    collector._on_button(_mouse_event(ax, MouseButton.MIDDLE))
    collector._on_button(_mouse_event(ax, MouseButton.LEFT, 3.0, 4.0))

    assert collector.point_sets == [[(1.0, 2.0)], [(3.0, 4.0)]]
    assert not collector._finished
    first_color = collector.point_artists[0][0].get_color()
    second_color = collector.point_artists[1][0].get_color()
    assert first_color != second_color
    plt.close(fig)


def test_middle_click_ignores_an_empty_set():
    fig, ax = plt.subplots()
    collector = _PointSetCollector(ax)

    collector._on_button(_mouse_event(ax, MouseButton.MIDDLE))

    assert collector.point_sets == [[]]
    plt.close(fig)


def test_right_click_removes_last_point_from_current_set():
    fig, ax = plt.subplots()
    collector = _PointSetCollector(ax)
    collector._on_button(_mouse_event(ax, MouseButton.LEFT))

    collector._on_button(_mouse_event(ax, MouseButton.RIGHT))

    assert collector.point_sets == [[]]
    assert collector.point_artists == [[]]
    plt.close(fig)


def test_explicit_set_count_finishes_on_last_middle_click():
    fig, ax = plt.subplots()
    fig.canvas.stop_event_loop = Mock()
    collector = _PointSetCollector(ax, n=2)
    collector._on_button(_mouse_event(ax, MouseButton.LEFT))
    collector._on_button(_mouse_event(ax, MouseButton.MIDDLE))
    collector._on_button(_mouse_event(ax, MouseButton.LEFT, 3.0, 4.0))

    collector._on_button(_mouse_event(ax, MouseButton.MIDDLE))

    assert collector._finished
    assert len(collector.point_sets) == 2
    fig.canvas.stop_event_loop.assert_called_once_with()
    plt.close(fig)


@pytest.mark.parametrize("key", ["enter", "return", "escape"])
def test_finish_key_stops_collection(key):
    fig, ax = plt.subplots()
    fig.canvas.stop_event_loop = Mock()
    collector = _PointSetCollector(ax)

    collector._on_key(SimpleNamespace(key=key))

    assert collector._finished
    fig.canvas.stop_event_loop.assert_called_once_with()
    plt.close(fig)


def test_convert_point_sets_calibrates_each_set():
    point_sets = [
        np.asarray([[0.0, 10.0], [5.0, 5.0]]),
        np.asarray([[10.0, 0.0]]),
    ]

    converted = _convert_point_sets(
        point_sets, left=0.0, right=10.0, bottom=10.0, top=0.0,
        xl=0.0, xr=100.0, yb=0.0, yt=1.0, xla=False, yla=False,
    )

    np.testing.assert_allclose(converted[0][0], [0.0, 50.0])
    np.testing.assert_allclose(converted[0][1], [0.0, 0.5])
    np.testing.assert_allclose(converted[1][0], [100.0])
    np.testing.assert_allclose(converted[1][1], [1.0])


def test_select_axis_corners_uses_two_diagonal_clicks(monkeypatch):
    fig, ax = plt.subplots()
    clicks = iter([[(10.0, 20.0)], [(110.0, 220.0)]])
    ginput = Mock(side_effect=lambda *args, **kwargs: next(clicks))
    monkeypatch.setattr(plt, "ginput", ginput)

    corners = _select_axis_corners(ax)

    assert corners == (10.0, 110.0, 220.0, 20.0)
    assert ginput.call_count == 2
    assert len(ax.lines) == 4
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [20.0, 20.0])
    np.testing.assert_allclose(ax.lines[1].get_xdata(), [10.0, 10.0])
    np.testing.assert_allclose(ax.lines[2].get_ydata(), [220.0, 220.0])
    np.testing.assert_allclose(ax.lines[3].get_xdata(), [110.0, 110.0])
    plt.close(fig)


def test_load_trace_image_uses_matplotlib_for_raster(monkeypatch):
    expected = np.ones((2, 3, 4))
    imread = Mock(return_value=expected)
    trace_curve_module = importlib.import_module("inpythotools.trace_curve")
    monkeypatch.setattr(trace_curve_module.mpimg, "imread", imread)

    image = _load_trace_image("plot.png")

    assert image is expected
    imread.assert_called_once_with("plot.png")


def test_load_trace_image_renders_selected_pdf_page(monkeypatch):
    pixmap = SimpleNamespace(
        samples=bytes(range(6)), height=1, width=2, n=3
    )
    pdf_page = MagicMock()
    pdf_page.get_pixmap.return_value = pixmap
    document = MagicMock()
    document.__enter__.return_value = document
    document.page_count = 3
    document.load_page.return_value = pdf_page
    color_space = object()
    pymupdf = SimpleNamespace(
        open=Mock(return_value=document), csRGB=color_space
    )
    monkeypatch.setitem(sys.modules, "pymupdf", pymupdf)

    image = _load_trace_image("plot.PDF", page=2, dpi=144)

    pymupdf.open.assert_called_once_with("plot.PDF")
    document.load_page.assert_called_once_with(1)
    pdf_page.get_pixmap.assert_called_once_with(
        dpi=144, colorspace=color_space, alpha=False
    )
    np.testing.assert_array_equal(
        image, np.asarray([[[0, 1, 2], [3, 4, 5]]], dtype=np.uint8)
    )


def test_load_trace_image_rejects_pdf_page_out_of_range(monkeypatch):
    document = MagicMock()
    document.__enter__.return_value = document
    document.page_count = 1
    pymupdf = SimpleNamespace(
        open=Mock(return_value=document), csRGB=object()
    )
    monkeypatch.setitem(sys.modules, "pymupdf", pymupdf)

    with pytest.raises(ValueError, match="document has 1 page"):
        _load_trace_image("plot.pdf", page=2)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [("page", 0, "page"), ("page", True, "page"),
     ("dpi", 0, "dpi"), ("dpi", 72.5, "dpi")],
)
def test_load_trace_image_validates_pdf_options(
        keyword, value, message):
    options = {keyword: value}
    with pytest.raises(ValueError, match=message):
        _load_trace_image("plot.pdf", **options)


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_invalid_set_count_is_rejected(n):
    with pytest.raises(ValueError):
        _validate_set_count(n)


@pytest.mark.parametrize("n", [None, 1, 3])
def test_valid_set_count_is_accepted(n):
    _validate_set_count(n)

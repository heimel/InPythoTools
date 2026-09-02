import pandas as pd
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication

import inpythotools.database_browser as database_browser
from inpythotools.database_browser import DatabaseBrowser


def test_suggested_export_filename_appends_exported_before_extension(tmp_path):
    source = tmp_path / "database.mat"

    assert database_browser._suggested_export_filename(source) == (
        tmp_path / "database_exported.mat"
    )


def test_database_browser_uses_bundled_24_px_icons():
    app = QApplication.instance() or QApplication([])
    window = DatabaseBrowser(pd.DataFrame())

    buttons = {
        button.accessibleName(): button
        for button in window.findChildren(type(window.save_button))
        if button.accessibleName()
    }
    for label in (
        "Load database",
        "Save database",
        "Close figures",
        "Apply filter",
        "Export database",
        "Explore session folder",
        "Settings",
        "First record",
        "Previous record",
        "Next record",
        "Last record",
        "Delete record",
        "Duplicate record",
    ):
        button = buttons[label]
        assert button is not None
        assert button.text() == ""
        assert not button.icon().isNull()
        assert button.toolTip() == label
        assert button.accessibleName() == label
        assert button.iconSize() == QSize(24, 24)
        assert button.width() == button.height()

    window.close()
    app.processEvents()


def test_filter_button_switches_between_apply_edit_and_disable_states():
    app = QApplication.instance() or QApplication([])
    window = DatabaseBrowser(
        pd.DataFrame([{"subject": "mouse-1"}, {"subject": "mouse-2"}])
    )
    button = window.filter_button

    assert button.accessibleName() == "Apply filter"
    assert button.toolTip() == "Apply filter"

    window.filter_box.setText("subject == 'mouse-1'")
    button.click()
    assert window.filtered_index == [0]
    assert button.accessibleName() == "Disable filter"
    assert button.toolTip() == "Disable filter"

    window.filter_box.setText("subject == 'mouse-2'")
    assert window.filtered_index == [0]
    assert button.accessibleName() == "Apply filter"
    assert button.toolTip() == "Apply filter"

    button.click()
    assert window.filtered_index == [1]
    assert button.accessibleName() == "Disable filter"
    assert window._has_active_filter()

    button.click()
    assert window.filter_box.text() == "subject == 'mouse-2'"
    assert window.filtered_index == [0, 1]
    assert button.accessibleName() == "Apply filter"
    assert not window._has_active_filter()

    button.click()
    assert window.filtered_index == [1]
    assert button.accessibleName() == "Disable filter"

    window.close()
    app.processEvents()


def test_database_browser_text_is_bold():
    app = QApplication.instance() or QApplication([])
    window = DatabaseBrowser(pd.DataFrame([{"subject": "mouse-1"}]))

    assert window.font().bold()
    assert window.filter_box.font().bold()
    assert window.status_label.font().bold()
    assert window.table.font().bold()
    assert window.table.item(0, 0).font().bold()
    assert not window.table.item(0, 1).font().bold()

    window.close()
    app.processEvents()


def test_button_rows_use_half_the_configured_horizontal_spacing():
    app = QApplication.instance() or QApplication([])
    window = DatabaseBrowser(pd.DataFrame(), spacing=6)
    layout = window.centralWidget().layout()

    assert layout.spacing() == 6
    assert layout.itemAt(0).layout().spacing() == 3
    assert layout.itemAt(1).layout().spacing() == 6
    assert layout.itemAt(2).layout().spacing() == 3

    window.close()
    app.processEvents()


def test_settings_button_opens_local_config(monkeypatch):
    app = QApplication.instance() or QApplication([])
    calls = []
    monkeypatch.setattr(database_browser, "edit_local_config", lambda: calls.append(True))
    window = DatabaseBrowser(pd.DataFrame())

    settings_button = next(
        button
        for button in window.findChildren(type(window.save_button))
        if button.accessibleName() == "Settings"
    )
    settings_button.click()

    assert calls == [True]
    assert settings_button.toolTip() == "Settings"

    window.close()
    app.processEvents()


def test_action_icons_are_optional_and_use_bundled_assets():
    app = QApplication.instance() or QApplication([])
    window = DatabaseBrowser(
        pd.DataFrame(),
        actions={"Analyze": lambda _record: None},
        action_icons={"Analyze": "microscope"},
    )

    analyze_button = next(
        button
        for button in window.findChildren(type(window.save_button))
        if button.accessibleName() == "Analyze"
    )
    assert analyze_button.text() == ""
    assert analyze_button.toolTip() == "Analyze"
    assert analyze_button.iconSize() == QSize(24, 24)
    assert not analyze_button.icon().isNull()

    window.close()
    app.processEvents()

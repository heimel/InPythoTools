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
        "Clear filter",
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


def test_database_browser_text_is_bold():
    app = QApplication.instance() or QApplication([])
    window = DatabaseBrowser(pd.DataFrame([{"subject": "mouse-1"}]))

    assert window.font().bold()
    assert window.filter_box.font().bold()
    assert window.status_label.font().bold()
    assert window.table.font().bold()
    assert window.table.item(0, 0).font().bold()
    assert window.table.item(0, 1).font().bold()

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

import pandas as pd
from PyQt6.QtWidgets import QApplication

import inpythotools.database_browser as database_browser
from inpythotools.database_browser import DatabaseBrowser


def test_compact_standard_icon_buttons():
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
    ):
        button = buttons[label]
        assert button is not None
        assert button.text() == ""
        assert not button.icon().isNull()
        assert button.toolTip() == label
        assert button.accessibleName() == label
        assert button.width() == button.height()

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
        if button.text() == "Settings"
    )
    settings_button.click()

    assert calls == [True]
    assert settings_button.toolTip() == "Edit local configuration"

    window.close()
    app.processEvents()

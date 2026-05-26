from __future__ import annotations


def test_novitrack_namespace_exports_core_functions() -> None:
    import novitrack as nt

    assert nt.analyse_nttestrecord.__name__ == "analyse_nttestrecord"
    assert nt.results_nttestrecord.__name__ == "results_nttestrecord"
    assert nt.browse_nt_database.__name__ == "browse_nt_database"


def test_inpythotools_exports_generic_browser() -> None:
    from inpythotools import browse_database

    assert browse_database.__name__ == "browse_database"

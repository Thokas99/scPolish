from importlib.metadata import version

import scpolish as scp


def test_import_and_version():
    assert scp.__version__ == version("scpolish")
    assert callable(scp.embedding)

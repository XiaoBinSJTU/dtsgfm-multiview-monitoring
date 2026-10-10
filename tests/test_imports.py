"""Public package import checks."""


def test_public_imports() -> None:
    from dtsgfm import DTSGFM, DTSGFMOptions, load_demo_config, total_loss

    assert DTSGFM is not None
    assert DTSGFMOptions is not None
    assert load_demo_config is not None
    assert total_loss is not None


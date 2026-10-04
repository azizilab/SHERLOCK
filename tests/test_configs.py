"""The configuration keys used to read perturbation and condition labels."""

import sherlock as slk


def test_set_and_reset_config():
    slk.configs.reset_config()
    default = slk.configs.get_config("ntc_label")
    slk.configs.set_config("ntc_label", "non-targeting")
    assert slk.configs.get_config("ntc_label") == "non-targeting"
    slk.configs.reset_config()
    assert slk.configs.get_config("ntc_label") == default

from __future__ import annotations

import pytest

from configuration.dataset             import InputConfig, Representation
from pipelines.dual.training.pipeline  import IfgChannelMap


def test_full_stack_four_tracks_maps_trailing_ifg_block():
    input_config = InputConfig.full_stack()

    assert IfgChannelMap.resolve(input_config, in_channels=9) == (5, 6, 7, 8)


def test_primary_and_ifgs_only_maps_after_primary():
    input_config = InputConfig()

    assert IfgChannelMap.resolve(input_config, in_channels=5) == (1, 2, 3, 4)


def test_dem_channel_stays_outside_the_ifg_block():
    input_config         = InputConfig.full_stack()
    input_config.use_dem = True

    assert IfgChannelMap.resolve(input_config, in_channels=10) == (5, 6, 7, 8)


def test_multi_channel_ifg_representation_expands_the_block():
    input_config = InputConfig.full_stack()
    input_config.interferograms_representation = Representation.MAG_ANGLE

    assert IfgChannelMap.resolve(input_config, in_channels=7) == (3, 4, 5, 6)


def test_disabled_interferograms_rejected():
    input_config = InputConfig.full_stack()
    input_config.use_interferograms = False

    with pytest.raises(ValueError, match="use_interferograms"):
        IfgChannelMap.resolve(input_config, in_channels=5)


def test_inconsistent_channel_count_rejected():
    input_config = InputConfig.full_stack()

    with pytest.raises(ValueError, match="unique track count"):
        IfgChannelMap.resolve(input_config, in_channels=2)

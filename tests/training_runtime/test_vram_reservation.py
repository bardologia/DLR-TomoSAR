from __future__ import annotations

import pytest
import torch

from configuration.training.general.pretraining import PretrainConfig
from configuration.training.general.runtime     import MemoryConfig
from tools.training.vram_reservation            import VramReservation

GB = 1024 ** 3
MB = 1024 ** 2

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _free_bytes() -> int:
    free, _total = torch.cuda.mem_get_info()
    return free


def _reservation(keep_free_gb: float, logger, enabled: bool = True) -> VramReservation:
    return VramReservation(enabled=enabled, keep_free_gb=keep_free_gb, device=torch.device("cuda"), logger=logger)


def test_memory_config_adopts_pretrain_values():
    pretrain = PretrainConfig(reserve_vram=True, vram_keep_free_gb=2.5)
    memory   = MemoryConfig()

    memory.adopt_reservation(pretrain)

    assert memory.reserve_vram      is True
    assert memory.vram_keep_free_gb == 2.5


def test_disabled_flag_is_inert(logger):
    reservation = VramReservation(enabled=False, keep_free_gb=1.0, device=torch.device("cuda"), logger=logger)

    reservation.fill()
    reservation.refill()

    assert reservation.enabled is False
    assert reservation.filled  is False


def test_cpu_device_disables_reservation(logger):
    reservation = VramReservation(enabled=True, keep_free_gb=1.0, device=torch.device("cpu"), logger=logger)

    reservation.fill()

    assert reservation.enabled is False
    assert reservation.filled  is False


@cuda_only
def test_fill_parks_memory_and_training_allocations_reuse_it(logger):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    free_before = _free_bytes()
    if free_before < 1 * GB:
        pytest.skip("not enough free VRAM for a safe reservation test")

    keep_free   = (free_before - 384 * MB) / GB
    reservation = _reservation(keep_free, logger)

    reserved_before = torch.cuda.memory_reserved()
    reservation.fill()

    assert reservation.filled is True
    assert _free_bytes() <= keep_free * GB + 64 * MB
    assert torch.cuda.memory_reserved() - reserved_before >= 256 * MB

    free_parked = _free_bytes()
    tensor      = torch.empty(128 * MB, dtype=torch.uint8, device="cuda")

    assert free_parked - _free_bytes() <= 32 * MB

    del tensor
    torch.cuda.empty_cache()


@cuda_only
def test_refill_reparks_after_cache_clear(logger):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    free_before = _free_bytes()
    if free_before < 1 * GB:
        pytest.skip("not enough free VRAM for a safe reservation test")

    keep_free   = (free_before - 384 * MB) / GB
    reservation = _reservation(keep_free, logger)

    reservation.fill()
    torch.cuda.empty_cache()

    assert _free_bytes() >= keep_free * GB + 256 * MB

    reservation.refill()

    assert _free_bytes() <= keep_free * GB + 64 * MB

    torch.cuda.empty_cache()


@cuda_only
def test_refill_before_fill_is_noop(logger):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    free_before = _free_bytes()
    reservation = _reservation(free_before / GB, logger)

    reservation.refill()

    assert reservation.filled is False
    assert abs(_free_bytes() - free_before) <= 16 * MB

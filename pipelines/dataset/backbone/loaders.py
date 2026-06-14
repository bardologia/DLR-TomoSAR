from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader, Dataset

from tools.monitoring.logger      import Logger
from tools.runtime.reproducibility import Reproducibility


class Loader:
    @staticmethod
    def build(
        train_dataset : Dataset,
        val_dataset   : Dataset,
        test_dataset  : Dataset,
        batch_size    : int,
        num_workers   : int,
        logger        : Logger,
        pin_memory    : bool = True,
        shuffle_train : bool = True,
        seed          : int  = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:

        logger.section("[Loaders]")
        logger.kv_table({
            "Batch size":    batch_size,
            "Num workers":   num_workers,
            "Pin memory":    pin_memory,
            "Shuffle train": shuffle_train,
            "Seed":          seed,
        })

        worker_init = Reproducibility.worker_init(seed) if num_workers > 0 else None

        _base = dict(
            batch_size         = batch_size,
            num_workers        = num_workers,
            pin_memory         = pin_memory,
            persistent_workers = num_workers > 0,
            prefetch_factor    = 8 if num_workers > 0 else None,
            worker_init_fn     = worker_init,
        )

        train_loader = DataLoader(train_dataset, shuffle = shuffle_train, drop_last = True,  generator = Reproducibility.generator(seed), **_base)
        val_loader   = DataLoader(val_dataset,   shuffle = False,         drop_last = False, **_base)
        test_loader  = DataLoader(test_dataset,  shuffle = False,         drop_last = False, **_base)

        logger.kv_table({
            "Train batches": len(train_loader),
            "Val batches":   len(val_loader),
            "Test batches":  len(test_loader),
        })

        return train_loader, val_loader, test_loader

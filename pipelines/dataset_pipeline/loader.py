from __future__ import annotations

from typing import Tuple

from torch.utils.data                    import DataLoader
from pipelines.dataset_pipeline.dataset  import PatchDataset
from tools.logger                        import Logger


class Loader:
    @staticmethod
    def build(
        train_dataset : PatchDataset,
        val_dataset   : PatchDataset,
        test_dataset  : PatchDataset,
        batch_size    : int,
        num_workers   : int,
        logger        : Logger,
        pin_memory    : bool = True,
        shuffle_train : bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:

        logger.section("[Loaders]")
        logger.kv_table({
            "Batch size":    batch_size,
            "Num workers":   num_workers,
            "Pin memory":    pin_memory,
            "Shuffle train": shuffle_train,
        })

        _base = dict(
            batch_size         = batch_size,
            num_workers        = num_workers,
            pin_memory         = pin_memory,
            persistent_workers = num_workers > 0,
            prefetch_factor    = 8 if num_workers > 0 else None,
        )

        train_loader = DataLoader(train_dataset, shuffle = shuffle_train, drop_last  = True, **_base,)
        val_loader   = DataLoader(val_dataset,   shuffle = False,         drop_last  = False, **_base,)
        test_loader  = DataLoader(test_dataset,  shuffle = False,         drop_last  = False, **_base,)

        logger.kv_table({
            "Train batches": len(train_loader),
            "Val batches":   len(val_loader),
            "Test batches":  len(test_loader),
        })

        return train_loader, val_loader, test_loader

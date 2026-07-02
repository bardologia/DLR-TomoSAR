import torch


class MetricAggregator:
    def __init__(self):
        self.components_sum : dict = {}
        self.weighted_sum   : dict = {}
        self.monitor_sum    : dict = {}
        self.occupancy_sum  : dict = {}
        self.physical_sum   : dict = {}
        self.extra_sum      : dict = {}
        self.count                = 0

    @staticmethod
    def _accumulate(store: dict, values: dict) -> None:
        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            store[k] = store.get(k, 0.0) + v

    def add(self, loss_dict: dict) -> None:
        self._accumulate(self.components_sum, loss_dict["components"])
        self._accumulate(self.weighted_sum,   loss_dict["weighted"])
        self._accumulate(self.monitor_sum,    loss_dict["monitor"])
        self._accumulate(self.occupancy_sum,  loss_dict["occupancy"])
        self._accumulate(self.physical_sum,   loss_dict["physical"])

        self.count += 1

    def add_extra(self, extra: dict) -> None:
        for k, v in extra.items():
            self.extra_sum[k] = self.extra_sum.get(k, 0.0) + (v if v == v else 0.0)

    def _reduce(self, store: dict) -> dict:
        n = max(1, self.count)
        return {k: float(v) / n for k, v in store.items()}

    def reduce_components(self) -> dict:
        return self._reduce(self.components_sum)

    def reduce_weighted(self) -> dict:
        return self._reduce(self.weighted_sum)

    def reduce_monitor(self) -> dict:
        return self._reduce(self.monitor_sum)

    def reduce_occupancy(self) -> dict:
        return self._reduce(self.occupancy_sum)

    def reduce_physical(self) -> dict:
        return self._reduce(self.physical_sum)

    def reduce_extra(self) -> dict:
        return self._reduce(self.extra_sum)

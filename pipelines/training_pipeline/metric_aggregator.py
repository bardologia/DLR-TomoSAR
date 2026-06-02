from __future__ import annotations


class MetricAggregator:
    def __init__(self):
        self.components_sum: dict = {}
        self.weighted_sum:   dict = {}
        self.extra_sum:      dict = {}
        self.count                = 0

    def add(self, loss_dict: dict) -> None:
        for k, v in loss_dict["components"].items():
            self.components_sum[k] = self.components_sum.get(k, 0.0) + float(v)

        for k, v in loss_dict["weighted"].items():
            self.weighted_sum[k] = self.weighted_sum.get(k, 0.0) + float(v)

        self.count += 1

    def add_extra(self, extra: dict) -> None:
        for k, v in extra.items():
            self.extra_sum[k] = self.extra_sum.get(k, 0.0) + (v if v == v else 0.0)

    def reduce_components(self) -> dict:
        n = max(1, self.count)
        return {k: v / n for k, v in self.components_sum.items()}

    def reduce_weighted(self) -> dict:
        n = max(1, self.count)
        return {k: v / n for k, v in self.weighted_sum.items()}

    def reduce_extra(self) -> dict:
        n = max(1, self.count)
        return {k: v / n for k, v in self.extra_sum.items()}

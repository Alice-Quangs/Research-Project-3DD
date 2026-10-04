"""Fixed bin catalogue and shared mutable packing state.

Importing this module does not read data or regenerate bins. All models and
heuristics must import this catalogue. Dimensions are mm; volumes are mm^3;
max_weight is kg. Costs use the agreed 143/232/317/1000 catalogue.

Heuristics own placement and EMS/EP updates. After modifying placed_items,
ems_list and/or ep_list, call register_placement(l, w, h) exactly once.
Weight, stability and blocking constraints must be enforced by the heuristic;
storing a capacity here does not enforce it.
"""

bin_types = [
    {"name": "small", "length": 462, "width": 304, "height": 546,
     "cost": 143, "max_weight": 22},
    {"name": "medium", "length": 745, "width": 325, "height": 399,
     "cost": 232, "max_weight": 14},
    {"name": "large", "length": 635, "width": 441, "height": 413,
     "cost": 317, "max_weight": 26},
    {"name": "extra_large", "length": 509, "width": 420, "height": 1260,
     "cost": 1000, "max_weight": 15},
]


class Bin:
    """Shared bin state for EMS and extreme-point heuristics.

    EMS entries: (x1, y1, z1, x2, y2, z2).
    EP entries: (x, y, z). The EP heuristic must maintain these points.
    Placed items: (x1, y1, z1, x2, y2, z2).
    """

    def __init__(self, bin_type):
        self.name = bin_type["name"]
        self.length = bin_type["length"]
        self.width = bin_type["width"]
        self.height = bin_type["height"]
        self.cost = bin_type["cost"]
        self.max_weight = bin_type.get("max_weight")
        if min(self.length, self.width, self.height) <= 0:
            raise ValueError("Bin dimensions must be positive.")
        self.volume = self.length * self.width * self.height
        self.used_volume = 0
        self.placed_items = []
        self.ems_list = [(0, 0, 0, self.length, self.width, self.height)]
        self.ep_list = [(0, 0, 0)]
        self.refresh_ems_stats()
        self.refresh_ep_stats()

    @property
    def residual_volume(self):
        return max(self.volume - self.used_volume, 0)

    def refresh_ems_stats(self):
        """Cache statistics after EMS updates, not for every prediction.

        max_ems_length/width/height describe the SAME largest-volume EMS.
        They are not independent axis maxima. Ties use the first EMS.
        EMS volumes may overlap: their sum is not residual bin volume.
        """
        dims = [(x2-x1, y2-y1, z2-z1)
                for x1, y1, z1, x2, y2, z2 in self.ems_list]
        volumes = [l*w*h for l, w, h in dims]
        self.ems_count = len(volumes)
        if volumes:
            max_i = max(range(len(volumes)), key=volumes.__getitem__)
            min_i = min(range(len(volumes)), key=volumes.__getitem__)
            self.max_ems_volume = volumes[max_i]
            self.min_ems_volume = volumes[min_i]
            self.mean_ems_volume = sum(volumes) / len(volumes)
            self.max_ems_length, self.max_ems_width, self.max_ems_height = dims[max_i]
            self.min_ems_length, self.min_ems_width, self.min_ems_height = dims[min_i]
            self.max_ems_dim = max(max(d) for d in dims)
        else:
            self.max_ems_volume = self.min_ems_volume = self.mean_ems_volume = 0
            self.max_ems_length = self.max_ems_width = self.max_ems_height = 0
            self.min_ems_length = self.min_ems_width = self.min_ems_height = 0
            self.max_ems_dim = 0
        # Compatibility with older scripts; not in the new feature schema.
        # This counts floor-level spaces, NOT all stable placements.
        self.supportable_ems_count = sum(ems[2] == 0 for ems in self.ems_list)
        self.ems_stats = {
            "num_ems": self.ems_count,
            "max_ems_volume": self.max_ems_volume,
            "mean_ems_volume": self.mean_ems_volume,
            "max_ems_dim": self.max_ems_dim,
            "supportable_ems_count": self.supportable_ems_count,
        }

    def refresh_ep_stats(self):
        """Cache summaries of the current EP list; empty-list summaries are 0."""
        self.number_eps = len(self.ep_list)
        for axis, index in (("x", 0), ("y", 1), ("z", 2)):
            values = [point[index] for point in self.ep_list]
            setattr(self, f"min_eps_{axis}", min(values) if values else 0)
            setattr(self, f"mean_eps_{axis}",
                    sum(values) / len(values) if values else 0)

    def register_placement(self, length, width, height):
        """Account for one placement after the heuristic updates its state.

        This does not append the item or generate EMSs/EPs. An EMS heuristic
        only needs refresh_ems_stats; an EP heuristic only needs
        refresh_ep_stats. The inactive representation is not maintained.
        """
        if min(length, width, height) <= 0:
            raise ValueError("Item dimensions must be positive.")
        self.used_volume += length * width * height
        self.refresh_ems_stats()
        self.refresh_ep_stats()

    def __repr__(self):
        return (f"Bin({self.name}, items={len(self.placed_items)}, "
                f"EMSs={len(self.ems_list)}, EPs={len(self.ep_list)})")

from __future__ import annotations

import numpy as np


class OccupancyGrid:
    """
    3D voxel grid representing occupied space inside a container.

    Each cell holds 0 (free) or a positive item index (occupied).
    All dimensions are integer voxel units; callers scale float dims
    by a precision factor before constructing the grid.
    """

    def __init__(self, L: int, W: int, H: int) -> None:
        self.L = L
        self.W = W
        self.H = H
        self._grid: np.ndarray = np.zeros((L, W, H), dtype=np.int32)
        self._item_counter: int = 1
        self._id_to_index: dict[str, int] = {}
        self._occupancy_prefix: np.ndarray | None = None

    def _invalidate_prefix(self) -> None:
        self._occupancy_prefix = None

    def _ensure_prefix(self) -> np.ndarray:
        if self._occupancy_prefix is None:
            occupied = (self._grid > 0).astype(np.int64, copy=False)
            prefix = np.zeros((self.L + 1, self.W + 1, self.H + 1), dtype=np.int64)
            prefix[1:, 1:, 1:] = occupied.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
            self._occupancy_prefix = prefix
        return self._occupancy_prefix

    def _occupied_count(self, x: int, y: int, z: int, l: int, w: int, h: int) -> int:
        prefix = self._ensure_prefix()
        x2 = x + l
        y2 = y + w
        z2 = z + h
        occupied = (
            prefix[x2, y2, z2]
            - prefix[x, y2, z2]
            - prefix[x2, y, z2]
            - prefix[x2, y2, z]
            + prefix[x, y, z2]
            + prefix[x, y2, z]
            + prefix[x2, y, z]
            - prefix[x, y, z]
        )
        return int(occupied)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_free(self, x: int, y: int, z: int, l: int, w: int, h: int) -> bool:
        """True iff the voxel region [x:x+l, y:y+w, z:z+h] is entirely free."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + l > self.L or y + w > self.W or z + h > self.H:
            return False
        return self._occupied_count(x, y, z, l, w, h) == 0

    def supported_area_fraction(self, x: int, y: int, z: int, l: int, w: int) -> float:
        """
        Fraction of the l×w base footprint at height z that is supported.

        z == 0 (container floor) → 1.0.
        Otherwise count occupied cells in the layer directly below (z-1).
        """
        if z == 0:
            return 1.0
        base_cells = l * w
        if base_cells == 0:
            return 0.0
        if x < 0 or y < 0 or z < 0:
            return 0.0
        if x + l > self.L or y + w > self.W or z > self.H:
            return 0.0
        supported = self._occupied_count(x, y, z - 1, l, w, 1)
        return supported / base_cells

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def place(self, item_id: str, x: int, y: int, z: int, l: int, w: int, h: int) -> None:
        """Mark the region as occupied by item_id."""
        idx = self._item_counter
        self._id_to_index[item_id] = idx
        self._grid[x : x + l, y : y + w, z : z + h] = idx
        self._item_counter += 1
        self._invalidate_prefix()

    def remove(self, item_id: str) -> None:
        """Free all voxels occupied by item_id."""
        idx = self._id_to_index.pop(item_id, None)
        if idx is not None:
            self._grid[self._grid == idx] = 0
            self._invalidate_prefix()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def copy(self) -> OccupancyGrid:
        """Return an independent deep copy (used for GA chromosome evaluation)."""
        clone = OccupancyGrid(self.L, self.W, self.H)
        clone._grid = self._grid.copy()
        clone._item_counter = self._item_counter
        clone._id_to_index = dict(self._id_to_index)
        if self._occupancy_prefix is not None:
            clone._occupancy_prefix = self._occupancy_prefix.copy()
        return clone

    def free_volume(self) -> int:
        return self.L * self.W * self.H - self.occupied_volume()

    def max_height(self) -> int:
        """Return the z-extent (in voxels) of the tallest occupied cell.
        Returns 0 if the grid is empty.
        """
        occupied_z = np.where(self._grid > 0)[2]
        if len(occupied_z) == 0:
            return 0
        return int(occupied_z.max()) + 1

    def occupied_volume(self) -> int:
        return self._occupied_count(0, 0, 0, self.L, self.W, self.H)

from pathlib import Path
from typing import Dict, Union

from poker_ai.ai.index import InfosetIndex, lmdb_map_size_for_players
from poker_ai.ai.regret_table import SparseRegretTable
from poker_ai.environment.game_state import PokerState

# Max number of abstract actions per betting round, derived from the action
# abstraction defined in RAISE_SIZES_BY_STAGE.  Built once at import time.
_MAX_ACTIONS_PER_STREET: Dict[int, int] = {
    r: len(PokerState.get_canonical_actions(r)) for r in range(4)
}


class Agent:
    """Training agent holding per-street sparse regret and strategy tables.

    All eight tables share one ``InfosetIndex`` so that the
    ``(chunk_id, row)`` coordinates returned by the index are valid for both
    the regret table and the corresponding strategy table on the same street.

    Parameters
    ----------
    index_path:
        Directory where the LMDB index is stored.  Created if absent.
    shm_dir:
        Directory for shared-memory chunk files.  Defaults to ``/dev/shm``.
        Override to a temporary directory in tests.
    lmdb_map_size:
        Maximum size of the LMDB environment in bytes.  The file is sparse on
        Linux so this is a reservation, not actual disk usage.  Defaults to
        ``None`` which uses the value of ``PLURIBUS_LMDB_MAP_SIZE`` env var
        or the module default (1 GiB for 2 players, 50 GiB for 3+).
        Use ``poker_ai.ai.index.lmdb_map_size_for_players()`` to compute a
        player-count-appropriate value.

    Attributes
    ----------
    regret_tables : Dict[int, SparseRegretTable]
        Per-street regret tables keyed by betting round (0–3).
    strategy_tables : Dict[int, SparseRegretTable]
        Per-street pre-flop strategy visit-count tables keyed by betting round
        (0–3).  Streets 1–3 remain empty until sub-game solving fires
        (Phase 5.7).
    """

    def __init__(
        self,
        index_path: Union[str, Path],
        shm_dir: str = "/dev/shm",
        lmdb_map_size: int = None,
    ) -> None:
        self._index = InfosetIndex(index_path, map_size=lmdb_map_size)
        self.regret_tables: Dict[int, SparseRegretTable] = {
            r: SparseRegretTable(
                n_actions=_MAX_ACTIONS_PER_STREET[r],
                table_name=f"pluribus_regret_{r}",
                index=self._index,
                shm_dir=shm_dir,
            )
            for r in range(4)
        }
        self.strategy_tables: Dict[int, SparseRegretTable] = {
            r: SparseRegretTable(
                n_actions=_MAX_ACTIONS_PER_STREET[r],
                table_name=f"pluribus_strategy_{r}",
                index=self._index,
                shm_dir=shm_dir,
            )
            for r in range(4)
        }

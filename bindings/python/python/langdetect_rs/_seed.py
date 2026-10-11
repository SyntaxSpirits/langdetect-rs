import random

from . import _core

_cache = (object(), None)


def seed_state(seed):
    """The Mersenne Twister state after `random.Random().seed(seed)`, or None for an OS seed.

    CPython does the seeding itself, so every seed type it accepts gives the same state.
    """
    global _cache
    if seed is None:
        return None
    key = (type(seed), seed)
    try:
        hash(key)
    except TypeError:
        key = None
    cached_key, state = _cache
    if key is not None and cached_key == key:
        return state
    rng = random.Random()
    rng.seed(seed)
    state = _core.SeedState(rng.getstate()[1])
    if key is not None:
        _cache = (key, state)
    return state

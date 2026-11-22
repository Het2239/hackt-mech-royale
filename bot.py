"""Mech Royale entrypoint. `mechfight.run` owns the stdin/stdout loop; we only build the brain and act."""
import os
import sys
import time

import mechfight as mf
from brain import Brain
from world import World


def find_map_dir(hello):
    m = hello.get("map") or {}
    base = m.get("dir") or "/arena"
    for f in m.get("files") or []:
        if f.endswith("world.json"):
            return os.path.dirname(os.path.join(base, f))
    for cand in (base, os.path.join(base, "v2"), "/arena/v2", "/arena"):
        if os.path.exists(os.path.join(cand, "world.json")):
            return cand
    return base


class Bot:
    name = "hackt"

    def __init__(self, hello: mf.Hello):
        t0 = time.perf_counter()
        world = World(find_map_dir(hello), hello["config"]["mech"])
        self.brain = Brain(hello, world)
        self.last = None
        print(f"ready: world ok={world.ok} in {time.perf_counter() - t0:.2f}s", file=sys.stderr, flush=True)

    def act(self, obs: mf.Observation) -> mf.Action:
        try:
            a = self.brain.act(obs)
            self.last = a
            return a
        except Exception as exc:  # a crash is a 0-point match; coast on the last move instead
            print(f"act error: {exc!r}", file=sys.stderr, flush=True)
            me = obs.get("self") or {}
            prev = self.last or {}
            return {
                "move": prev.get("move", [0.0, 0.0]),
                "boost": False,
                "aim": [float(v) for v in (me.get("aim") or [0.0, 0.0])],
                "fire": [False, False],
            }


mf.run(Bot)

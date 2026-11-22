"""Arena model: terrain heights, a coarse grid of solid tops for line of sight, and A* navigation.

Everything is loaded once in Bot.__init__ (inside the 20 s hello budget). Per-tick queries are cheap:
`los` samples a segment, `path` runs a bounded A* on the nav grid.
"""
import array
import heapq
import json
import math
import os
import time

CELL = 3.0  # nav / solid grid size, metres
LCELL = float(os.environ.get("BOT_LCELL", "1.0"))  # line-of-sight grid size, metres


def _scan_fill(ring, cell, x0, y0, nx, ny, visit):
    """Call visit(i, j) for every grid cell whose centre lies inside polygon `ring` (flat [x,y,...])."""
    pts = [(ring[k], ring[k + 1]) for k in range(0, len(ring) - 1, 2)]
    if len(pts) < 3:
        return
    ys = [p[1] for p in pts]
    j0 = max(0, int((min(ys) - y0) / cell))
    j1 = min(ny - 1, int((max(ys) - y0) / cell) + 1)
    n = len(pts)
    for j in range(j0, j1 + 1):
        cy = y0 + (j + 0.5) * cell
        xs = []
        for k in range(n):
            ax, ay = pts[k]
            bx, by = pts[(k + 1) % n]
            if (ay <= cy < by) or (by <= cy < ay):
                xs.append(ax + (cy - ay) * (bx - ax) / (by - ay))
        xs.sort()
        for a, b in zip(xs[0::2], xs[1::2]):
            i0 = max(0, int(math.ceil((a - x0) / cell - 0.5)))
            i1 = min(nx - 1, int(math.floor((b - x0) / cell - 0.5)))
            for i in range(i0, i1 + 1):
                visit(i, j)


class World:
    def __init__(self, map_dir, mech_cfg):
        self.ok = False
        self.step_up = float(mech_cfg.get("step_up", 1.2))
        try:
            self._load(map_dir)
            self.ok = True
        except Exception as exc:  # no map: the bot still plays, just without LOS/nav
            import sys
            print(f"world: map load failed: {exc}", file=sys.stderr, flush=True)

    # ------------------------------------------------------------------ loading
    def _load(self, map_dir):
        with open(os.path.join(map_dir, "world.json")) as f:
            w = json.load(f)
        t = w["terrain"]
        self.tx0, self.ty0, self.tstep = t["minX"], t["minY"], t["step"]
        self.tnx, self.tny, self.tscale = t["nx"], t["ny"], t["scale"]
        h = array.array("h")
        with open(os.path.join(map_dir, t["file"]), "rb") as f:
            h.frombytes(f.read())
        self.th = h

        b = w["bounds"]
        self.x0, self.y0 = b["minX"], b["minY"]
        self.nx = int((b["maxX"] - b["minX"]) / CELL) + 1
        self.ny = int((b["maxY"] - b["minY"]) / CELL) + 1
        n = self.nx * self.ny
        # top[k]: height of the highest solid over cell k (-1e9 = none); ground[k]: terrain at the centre
        self.ground = array.array("f", [0.0]) * n
        for j in range(self.ny):
            cy = self.y0 + (j + 0.5) * CELL
            for i in range(self.nx):
                self.ground[j * self.nx + i] = self.terrain(self.x0 + (i + 0.5) * CELL, cy)
        self.top = array.array("f", [-1e9]) * n
        nx, ny, top = self.nx, self.ny, self.top
        for p in w["prisms"]:
            if p["kind"] == "bridge":
                continue  # overhangs: walk and see under them
            pt = p["top"]

            def visit(i, j, pt=pt):
                k = j * nx + i
                if pt > top[k]:
                    top[k] = pt
            _scan_fill(p["ring"], CELL, self.x0, self.y0, nx, ny, visit)
        # a finer grid of solid tops for line of sight: 3 m cells lose corners and narrow gaps
        self.lnx = int((b["maxX"] - b["minX"]) / LCELL) + 1
        self.lny = int((b["maxY"] - b["minY"]) / LCELL) + 1
        lnx, lny = self.lnx, self.lny
        ltop = array.array("f", [-1e9]) * (lnx * lny)
        for p in w["prisms"]:
            if p["kind"] == "bridge":
                continue
            pt = p["top"]

            def lvisit(i, j, pt=pt):
                k = j * lnx + i
                if pt > ltop[k]:
                    ltop[k] = pt
            _scan_fill(p["ring"], LCELL, self.x0, self.y0, lnx, lny, lvisit)
        self.ltop = ltop
        # walls that block movement (fences); line-rasterised
        self.fence = bytearray(n)
        for br in w.get("barriers", []):
            if not br.get("blocks"):
                continue
            ln = br["line"]
            for k in range(0, len(ln) - 3, 2):
                ax, ay, bx, by = ln[k], ln[k + 1], ln[k + 2], ln[k + 3]
                steps = int(math.hypot(bx - ax, by - ay) / (CELL * 0.5)) + 1
                for s in range(steps + 1):
                    c = self.cell(ax + (bx - ax) * s / steps, ay + (by - ay) * s / steps)
                    if c is not None:
                        self.fence[c] = 1
        # walkable: no solid rising more than a step above the ground, then inflated by one cell for the body
        raw = bytearray(n)
        # terrain too steep to walk (retaining walls, embankments, the stadium bowl): any 1 m vertex step
        # above what step_up + max slope allow marks the containing nav cell
        lim = (self.step_up + 0.15) / self.tscale
        th, tnx, tny = self.th, self.tnx, self.tny
        for j in range(tny - 1):
            row = j * tnx
            cy = int((self.ty0 + j * self.tstep - self.y0) / CELL)
            if not 0 <= cy < ny:
                continue
            for i in range(tnx - 1):
                k = row + i
                h0 = th[k]
                if abs(th[k + 1] - h0) > lim or abs(th[k + tnx] - h0) > lim:
                    cx = int((self.tx0 + i * self.tstep - self.x0) / CELL)
                    if 0 <= cx < nx:
                        raw[cy * nx + cx] = 1
        for k in range(n):
            if self.fence[k] or top[k] > self.ground[k] + self.step_up + 0.3:
                raw[k] = 1
        self.blocked = bytearray(raw)
        for j in range(ny):
            for i in range(nx):
                if raw[j * nx + i]:
                    continue
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < nx and 0 <= jj < ny and raw[jj * nx + ii]:
                        self.blocked[j * nx + i] = 2  # soft: near a wall, walkable at a cost
                        break
        self.raw = raw

    # ------------------------------------------------------------------ queries
    def terrain(self, x, y):
        fx = (x - self.tx0) / self.tstep
        fy = (y - self.ty0) / self.tstep
        i = min(max(int(fx), 0), self.tnx - 2)
        j = min(max(int(fy), 0), self.tny - 2)
        ax, ay = min(max(fx - i, 0.0), 1.0), min(max(fy - j, 0.0), 1.0)
        h, nx = self.th, self.tnx
        k = j * nx + i
        a = h[k] * (1 - ax) + h[k + 1] * ax
        b = h[k + nx] * (1 - ax) + h[k + nx + 1] * ax
        return (a * (1 - ay) + b * ay) * self.tscale

    def cell(self, x, y):
        i = int((x - self.x0) / CELL)
        j = int((y - self.y0) / CELL)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return j * self.nx + i
        return None

    def centre(self, k):
        return (self.x0 + (k % self.nx + 0.5) * CELL, self.y0 + (k // self.nx + 0.5) * CELL)

    def walkable(self, x, y):
        if not self.ok:
            return True
        c = self.cell(x, y)
        return c is not None and self.raw[c] == 0

    def los(self, a, b, step=1.0):
        """True if the segment a->b (3D) clears every solid top and the terrain."""
        if not self.ok:
            return True
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        d = math.hypot(dx, dy)
        n = int(d / step)
        if n < 2:
            return True
        top, nx, ny, x0, y0 = self.ltop, self.lnx, self.lny, self.x0, self.y0
        for s in range(1, n):
            f = s / n
            x, y, z = a[0] + dx * f, a[1] + dy * f, a[2] + dz * f
            i = int((x - x0) / LCELL)
            j = int((y - y0) / LCELL)
            if 0 <= i < nx and 0 <= j < ny and top[j * nx + i] > z:
                return False
            if s % 3 == 0 and self.terrain(x, y) > z:
                return False
        return True

    def ray_clear(self, x, y, yaw, dist):
        """Free walking distance from (x, y) along yaw, up to dist."""
        if not self.ok:
            return dist
        c, s = math.cos(yaw), math.sin(yaw)
        d = 1.5
        while d <= dist:
            k = self.cell(x + c * d, y + s * d)
            if k is None or self.raw[k]:
                return d - 1.5
            d += 1.5
        return dist

    def nearest_free(self, x, y, r=6):
        c = self.cell(x, y)
        if c is None:
            return None
        if self.raw[c] == 0:
            return c
        i0, j0 = c % self.nx, c // self.nx
        for rr in range(1, r + 1):
            for di in range(-rr, rr + 1):
                for dj in (-rr, rr) if abs(di) != rr else range(-rr, rr + 1):
                    i, j = i0 + di, j0 + dj
                    if 0 <= i < self.nx and 0 <= j < self.ny and self.raw[j * self.nx + i] == 0:
                        return j * self.nx + i
        return None

    def path(self, sx, sy, gx, gy, budget_s=0.012, max_expand=6000):
        """A* from start to goal; returns a list of (x, y) waypoints or None. Bounded by time and expansions;
        on budget exhaustion returns the path to the closest node reached."""
        if not self.ok:
            return None
        s = self.nearest_free(sx, sy)
        g = self.nearest_free(gx, gy, 10)
        if s is None or g is None:
            return None
        nx, blocked = self.nx, self.blocked
        gi, gj = g % nx, g // nx
        def hfun(k):
            di, dj = abs(k % nx - gi), abs(k // nx - gj)
            return (max(di, dj) + 0.414 * min(di, dj))
        openq = [(hfun(s), 0.0, s)]
        came = {s: -1}
        cost = {s: 0.0}
        best, besth = s, hfun(s)
        t_end = time.perf_counter() + budget_s
        expanded = 0
        nbrs = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))
        ny = self.ny
        while openq:
            f, gc, k = heapq.heappop(openq)
            if k == g:
                best = g
                break
            if gc > cost.get(k, 1e18):
                continue
            expanded += 1
            if expanded > max_expand or (expanded & 255 == 0 and time.perf_counter() > t_end):
                break
            i, j = k % nx, k // nx
            for di, dj, w in nbrs:
                ii, jj = i + di, j + dj
                if not (0 <= ii < nx and 0 <= jj < ny):
                    continue
                kk = jj * nx + ii
                b = blocked[kk]
                if b == 1:
                    continue
                if di and dj and (blocked[j * nx + ii] == 1 or blocked[jj * nx + i] == 1):
                    continue  # no corner cutting
                nc = gc + w * (2.5 if b == 2 else 1.0)
                if nc < cost.get(kk, 1e18):
                    cost[kk] = nc
                    came[kk] = k
                    hh = hfun(kk)
                    if hh < besth:
                        best, besth = kk, hh
                    heapq.heappush(openq, (nc + hh, nc, kk))
        out = []
        k = best
        while k != -1:
            out.append(self.centre(k))
            k = came[k]
        out.reverse()
        return out

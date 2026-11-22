"""Decision core. One `act` per observation; every branch returns a complete action.

Policy (practice = us vs a 15-bot house squad, score = mechs that die before us):
  1. survive the zone (arrival-time check against the shrink clock)
  2. dodge mortar shells, keep out of lightning range of lightning carriers
  3. retreat below ~38 % HP to a spot hidden from known threats and hold fire so regen kicks in
  4. fight selectively: mechs that hunt/attack us, are close, or are nearly dead; strafe at a band per weapon
  5. snowball: scoop cubes (especially from our kills), grab a laser/lightning, capture supply drops when hurt
  6. otherwise rotate to a point inside the next circle
"""
import math
import random

TAU = 2 * math.pi
SECONDARY_PREF = {"laser": 3, "lightning": 2, "mortar": -1}  # never carry the mortar: cookoff + loud
UPGRADE_ORDER = [
    "plating", "armor", "laser_damage", "laser_charge", "nanites", "gun_rate", "gun_damage",
    "lightning_efficiency", "gun_mag", "lightning_range", "servos", "gun_velocity", "jets", "reactor",
    "capacitor", "gyros", "sensors", "laser_pierce", "lightning_chain",
]
BAND = {"laser": (55.0, 110.0), "lightning": (12.0, 30.0), "gun": (35.0, 75.0)}  # preferred fight range


P = {
    "retreat_hp": 0.38,      # start disengaging below this HP fraction
    "recover_hp": 0.85,      # stop disengaging above this
    "multi_retreat_hp": 0.65,  # disengage earlier when 2+ mechs are hitting us
    "engage_range": 70.0,    # fight anything this close even if it has not attacked us
    "hunt_hp": 0.6,          # chase a lost target only while we are above this HP fraction
    "hunt_s": 6.0,           # how long a lost target stays worth chasing
    "lead": 0.6,             # fraction of target velocity to lead the gun by
    "cautious_t": 265.0,     # after this match time, stop chasing and hold the circle
    "finish_ratio": 0.6,     # keep fighting while retreating if target hp < ratio * ours
}
try:
    import json as _json, os as _os
    P.update(_json.loads(_os.environ.get("BOT_PARAMS", "{}")))
except Exception:
    pass


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def d2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(yaw):
    return [math.cos(yaw), math.sin(yaw)]


class Brain:
    def __init__(self, hello, world):
        cfg = hello["config"]
        self.cfg = cfg
        self.world = world
        self.rng = random.Random(hello.get("seed", 0))
        m = cfg["mech"]
        self.walk = m.get("walk_speed", 11.7)
        self.boost_speed = m.get("boost_speed", 19.5)
        self.eye = m.get("eye_height", 8.0)
        self.muzzle = m.get("muzzle_height", 7.0)
        self.height = m.get("height", 9.0)
        self.reach = cfg["loot"]["pickup_radius"]
        self.cube_reach = cfg["cubes"].get("pickup_radius", 3.5)
        self.cube_cap = cfg["cubes"].get("stat_cap", 10)
        self.w = cfg["weapons"]
        self.my_id = hello.get("slot")
        # memory
        self.attackers = {}     # id -> last time they damaged us
        self.hit_bearing = None  # (t, yaw) of the most recent hit
        self.my_targets = {}     # id -> last time we damaged them
        self.kills = 0
        self.kill_spots = []     # (t, x, y): cubes drop where we kill
        self.last_seen = {}      # id -> (t, pos, hp, weapons)
        self.retreating = False
        self.strafe = 1.0
        self.strafe_until = 0.0
        self.path = None
        self.path_goal = None
        self.path_t = -9.0
        self.stuck_t = 0.0
        self.unstuck_until = 0.0
        self.unstuck_yaw = 0.0
        self.last_pos = None
        self.alive0 = None
        self.ffa = False
        self.zone_goal = None
        self.zone_key = None
        self.scan_yaw = 0.0
        self.target_id = None
        self.dead_ids = set()

    # ------------------------------------------------------------------ helpers
    def secondary(self, me):
        ws = me.get("weapons") or []
        return ws[1] if len(ws) > 1 else None

    def los_to(self, me, pos):
        a = (me["pos"][0], me["pos"][1], me["pos"][2] + self.eye)
        b = (pos[0], pos[1], pos[2] + self.height * 0.55)
        return self.world.los(a, b)

    def aim_at(self, me, tpos, tvel, speed=None, lead=0.6):
        """[yaw, pitch] from our muzzle to the target's mid-body, leading by `lead` of its velocity."""
        x, y, z = me["pos"]
        mz = z + self.muzzle
        tx, ty, tz = tpos[0], tpos[1], tpos[2] + self.height * 0.5
        if speed:
            for _ in range(2):
                tof = math.dist((x, y, mz), (tx, ty, tz)) / speed
                tx = tpos[0] + tvel[0] * tof * lead
                ty = tpos[1] + tvel[1] * tof * lead
        dxy = math.hypot(tx - x, ty - y)
        return [math.atan2(ty - y, tx - x), math.atan2(tz - mz, max(dxy, 0.1))]

    def mortar_aim(self, me, tpos):
        st = self.w["mortar"]
        v, g = st.get("speed", 72), st.get("gravity", 22)
        x, y, z = me["pos"]
        dxy = math.hypot(tpos[0] - x, tpos[1] - y)
        h = tpos[2] - (z + self.muzzle)
        disc = v ** 4 - g * (g * dxy * dxy + 2 * h * v * v)
        if disc < 0 or dxy < 1:
            return None
        pitch = math.atan((v * v - math.sqrt(disc)) / (g * dxy))
        return [math.atan2(tpos[1] - y, tpos[0] - x), pitch]

    def steer(self, me, yaw, look=7.0):
        """Rotate a desired heading away from walls; returns a unit move vector."""
        x, y = me["pos"][0], me["pos"][1]
        w = self.world
        if not w.ok or not w.walkable(x, y):
            return unit(yaw)
        for off in (0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05, 1.5, -1.5, 2.1, -2.1):
            if w.ray_clear(x, y, yaw + off, look) >= look - 0.1:
                return unit(yaw + off)
        return unit(yaw)

    def go(self, me, t, gx, gy):
        """Move toward (gx, gy) along a cached A* path."""
        x, y = me["pos"][0], me["pos"][1]
        if d2((x, y), (gx, gy)) < 3:
            return [0.0, 0.0]
        w = self.world
        if not w.ok:
            return self.steer(me, math.atan2(gy - y, gx - x))
        replan = (self.path is None or self.path_goal is None or d2(self.path_goal, (gx, gy)) > 10
                  or t - self.path_t > 2.0)
        if replan:
            p = w.path(x, y, gx, gy)
            self.path, self.path_goal, self.path_t = p, (gx, gy), t
        p = self.path
        if not p:
            return self.steer(me, math.atan2(gy - y, gx - x))
        while len(p) > 1 and d2(p[0], (x, y)) < 4.5:
            p.pop(0)
        # look ahead to the farthest of the next few waypoints we can walk straight to
        tgt = p[0]
        for q in p[1:6]:
            yaw = math.atan2(q[1] - y, q[0] - x)
            dist = d2(q, (x, y))
            if w.ray_clear(x, y, yaw, dist) >= dist - 1.0:
                tgt = q
            else:
                break
        if len(p) == 1:
            tgt = (gx, gy) if w.walkable(gx, gy) else p[0]
        return self.steer(me, math.atan2(tgt[1] - y, tgt[0] - x), look=4.0)

    def hidden_spot(self, me, threats, zone):
        """Nearest walkable point within ~40 m that no threat can see, preferring the zone."""
        x, y, z = me["pos"]
        best, bs = None, 1e9
        cx, cy = zone["center"]
        for r in (10.0, 18.0, 28.0, 40.0):
            for k in range(12):
                a = k * TAU / 12
                px, py = x + r * math.cos(a), y + r * math.sin(a)
                if not self.world.walkable(px, py):
                    continue
                pz = self.world.terrain(px, py)
                ok = True
                for tp in threats:
                    if self.world.los((tp[0], tp[1], tp[2] + self.eye), (px, py, pz + self.height * 0.6)):
                        ok = False
                        break
                if not ok:
                    continue
                s = r
                if math.hypot(px - cx, py - cy) > zone["radius"] - 10:
                    s += 80
                for tp in threats:
                    s -= 0.15 * min(d2((px, py), tp), 80)  # farther from threats is better
                if s < bs:
                    best, bs = (px, py), s
            if best is not None and r >= 18:
                break
        return best

    # ------------------------------------------------------------------ zone
    def zone_target(self, me, obs):
        z = obs["zone"]
        t = obs["t"]
        x, y = me["pos"][0], me["pos"][1]
        nc, nr = z["next_center"], z["next_radius"]
        key = (round(nc[0]), round(nc[1]), round(nr))
        if key != self.zone_key:
            self.zone_key = key
            # one goal per circle: a point well inside the next circle, on our side of it
            dx, dy = x - nc[0], y - nc[1]
            d = math.hypot(dx, dy)
            inner = max(nr * 0.55, nr - 60) if nr > 30 else nr * 0.3
            if t > P["cautious_t"]:
                inner = min(inner, nr * 0.35)  # late: hold near the centre of the closing circle
            if d <= inner:
                self.zone_goal = (x, y)
            else:
                self.zone_goal = (nc[0] + dx / d * inner, nc[1] + dy / d * inner)
        # urgency: will we be caught by the edge?
        dist_next = math.hypot(x - nc[0], y - nc[1]) - nr
        time_left = z["shrink_end"] - t
        need = max(0.0, dist_next) / self.walk
        cur_out = math.hypot(x - z["center"][0], y - z["center"][1]) > z["radius"] - 4
        urgent = cur_out or (dist_next > -5 and need + 12 > time_left)
        return self.zone_goal, urgent, cur_out

    # ------------------------------------------------------------------ main
    def act(self, obs):
        t = obs["t"]
        me = obs["self"]
        x, y, z = me["pos"]
        hpf = me["hp"] / max(me["max_hp"], 1)
        act = {"move": [0.0, 0.0], "boost": False, "fire": [False, False]}
        if self.alive0 is None:
            self.alive0 = obs.get("alive", 16)

        # ---- events
        for e in obs.get("events") or []:
            k = e.get("type")
            if k == "damaged" and e.get("by") is not None:
                self.attackers[e["by"]] = t
                if e.get("bearing") is not None:
                    self.hit_bearing = (t, e["bearing"])
            elif k == "hit":
                self.my_targets[e["target"]] = t
            elif k == "kill":
                self.kills += 1
                self.dead_ids.add(e["target"])
                ls = self.last_seen.get(e["target"])
                if ls:
                    self.kill_spots.append((t, ls[1][0], ls[1][1]))
                self.attackers.pop(e["target"], None)
                if self.target_id == e["target"]:
                    self.target_id = None
        # deaths we did not cause => free-for-all (practice squads never kill each other)
        others_dead = self.alive0 - obs.get("alive", self.alive0) - self.kills
        if others_dead >= 5:
            self.ffa = True

        mechs = [m for m in obs.get("mechs") or [] if m.get("kind") == "mech" and m["id"] not in self.dead_ids]
        npcs = [m for m in obs.get("mechs") or [] if m.get("kind") == "npc"]
        for m in mechs:
            self.last_seen[m["id"]] = (t, m["pos"], m.get("hp", 1400), m.get("weapons") or [])

        # ---- upgrades
        off = obs.get("offer")
        if off and off.get("choices"):
            ids = [c["id"] for c in off["choices"]]
            pick = 0
            for u in UPGRADE_ORDER:
                if u in ids:
                    pick = ids.index(u)
                    break
            act["pick"] = pick

        # ---- weapons on the ground
        sec = self.secondary(me)
        sec_kind = sec["kind"] if sec else None
        cur_pref = SECONDARY_PREF.get(sec_kind, 0) if sec_kind else 0
        for p in obs.get("pickups") or []:
            if d2(p["pos"], me["pos"]) < self.reach and SECONDARY_PREF.get(p["kind"], -2) > max(cur_pref, 0):
                act["take"] = p["id"]
                if sec is not None:
                    act["drop"] = 1
                break
        if sec_kind == "mortar":
            act["drop"] = 1  # it makes us a cookoff target

        # ---- threats
        def dist(m):
            return d2(m["pos"], me["pos"])
        visible = []
        for m in mechs:
            m["_d"] = dist(m)
            m["_los"] = m["_d"] < 300 and self.los_to(me, m["pos"])
            visible.append(m)
        hunters = [m for m in visible if t - self.attackers.get(m["id"], -99) < 8 or m["_d"] < 60
                   or m.get("charging")]
        lightning_threats = [m for m in visible if "lightning" in (m.get("weapons") or [])
                             and m["_d"] < self.w["lightning"]["range"] + 14]

        # ---- choose a target
        target = None
        best = 1e9
        for m in visible:
            if not m["_los"]:
                continue
            mhp = m.get("hp", 1400) / max(m.get("max_hp", 1400), 1)
            engage = (self.ffa or t - self.attackers.get(m["id"], -99) < 10 or m["_d"] < P["engage_range"]
                      or mhp < 0.35 or t - self.my_targets.get(m["id"], -99) < 6)
            if not engage or m["_d"] > 230:
                continue
            s = m["_d"] + 260 * mhp - (60 if m["id"] == self.target_id else 0)
            if s < best:
                best, target = s, m
        self.target_id = target["id"] if target else None

        # ---- retreat state machine (hysteresis), unless the target is nearly dead
        recent_attackers = [a for a, ta in self.attackers.items() if t - ta < 3 and a not in self.dead_ids]
        if hpf < P["retreat_hp"] or (len(recent_attackers) >= 2 and hpf < P["multi_retreat_hp"]):
            self.retreating = True
        elif hpf > P["recover_hp"]:
            self.retreating = False
        finishing = target is not None and target.get("hp", 1400) < P["finish_ratio"] * me["hp"] and target["_d"] < 90
        retreat = self.retreating and not finishing and (hunters or t - self.attackers_last() < 6)

        zgoal, urgent, outside = self.zone_target(me, obs)

        # ---- movement
        move_yaw = None
        boost = False
        mode = "rotate"
        shells = [p for p in obs.get("projectiles") or [] if p.get("kind") == "mortar"]
        dodge = self.mortar_dodge(me, shells)
        if t < self.unstuck_until:
            mode = "unstuck"
            act["move"] = unit(self.unstuck_yaw)
            act["jump"] = True
        elif outside or urgent:
            mode = "zone"
            act["move"] = self.go(me, t, *zgoal)
            boost = outside or urgent
        elif dodge is not None:
            mode = "dodge"
            act["move"] = self.steer(me, dodge, 5)
            boost = True
        elif lightning_threats and sec_kind != "lightning":
            mode = "anti-lightning"
            c = lightning_threats[0]
            away = math.atan2(y - c["pos"][1], x - c["pos"][0])
            act["move"] = self.steer(me, away + 0.4 * self.strafe)
            boost = True
        elif retreat:
            mode = "retreat"
            tps = [m["pos"] for m in hunters] or [m["pos"] for m in visible[:3]]
            spot = self.hidden_spot(me, tps, obs["zone"]) if tps else None
            if spot:
                act["move"] = self.go(me, t, *spot)
            else:
                if tps:
                    away = math.atan2(y - tps[0][1], x - tps[0][0])
                elif self.hit_bearing:
                    away = self.hit_bearing[1] + math.pi
                else:
                    away = math.atan2(zgoal[1] - y, zgoal[0] - x)
                act["move"] = self.steer(me, away)
            boost = True
        elif target is not None:
            mode = "fight"
            act["move"] = self.fight_move(me, t, target, sec_kind)
        elif self.hunt_target(me, t, hpf, obs) is not None:
            mode = "hunt"
            hp_ = self.hunt_target(me, t, hpf, obs)
            act["move"] = self.go(me, t, hp_[0], hp_[1])
        else:
            goal = self.loot_goal(me, obs, visible, hpf, sec_kind, t)
            if goal is not None:
                mode = "loot"
                act["move"] = self.go(me, t, goal[0], goal[1])
            else:
                act["move"] = self.go(me, t, *zgoal)

        # stuck detection (we asked to move but did not)
        mv = act["move"]
        if self.last_pos is not None and math.hypot(*mv) > 0.5 and mode != "unstuck":
            if math.hypot(x - self.last_pos[0], y - self.last_pos[1]) < 0.25:
                self.stuck_t += 0.1
            else:
                self.stuck_t = max(0.0, self.stuck_t - 0.05)
            if self.stuck_t > 1.2:
                self.stuck_t = 0.0
                self.unstuck_until = t + 1.0
                self.unstuck_yaw = math.atan2(mv[1], mv[0]) + self.rng.choice((-1.6, 1.6, math.pi))
                self.path = None
        self.last_pos = (x, y)
        if me["energy"] < 15:
            boost = False
        act["boost"] = boost

        # ---- aim and fire
        self.aim_and_fire(act, me, t, target, visible, npcs, obs, sec, retreat, mode)
        if me.get("grounded") and mode in ("zone", "retreat") and self.stuck_t > 0.6:
            act["jump"] = True
        try:
            import mechfight as mf
            mf.draw.label(f"{mode} hp{hpf:.2f} k{self.kills}{' ffa' if self.ffa else ''}")
        except Exception:
            pass
        return act

    def hunt_target(self, me, t, hpf, obs):
        """Last known position of a mech we were trading with, if worth chasing."""
        if hpf < P["hunt_hp"] or (t > P["cautious_t"] and not self.ffa):
            return None
        best, bs = None, 1e9
        nc, nr = obs["zone"]["next_center"], obs["zone"]["next_radius"]
        for mid, (ts, pos, hp, _w) in self.last_seen.items():
            if mid in self.dead_ids or t - ts > P["hunt_s"]:
                continue
            traded = t - self.my_targets.get(mid, -99) < P["hunt_s"] or t - self.attackers.get(mid, -99) < P["hunt_s"]
            if not traded or hp > me["hp"] * 1.1:
                continue
            if math.hypot(pos[0] - nc[0], pos[1] - nc[1]) > nr + 30:
                continue
            dd = d2(pos, me["pos"]) + 0.1 * hp
            if dd < bs:
                best, bs = pos, dd
        return best

    def cube_bonus(self, me):
        c = self.cfg["cubes"]
        return min(me.get("cubes", 0), c.get("stat_cap", 10)) * c.get("damage_per_cube", 0.04)

    def attackers_last(self):
        return max(self.attackers.values(), default=-99)

    def mortar_dodge(self, me, shells):
        """Heading perpendicular to an incoming mortar shell's ground track, or None."""
        x, y, z = me["pos"]
        g = self.w["mortar"].get("gravity", 22)
        r = self.w["mortar"].get("splash_radius", 7) + 4
        for p in shells:
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            # time to fall to our height
            a, b, c = -0.5 * g, vz, pz - z
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            tl = (-b - math.sqrt(disc)) / (2 * a)
            if tl <= 0 or tl > 3.5:
                continue
            lx, ly = px + vx * tl, py + vy * tl
            if math.hypot(lx - x, ly - y) < r:
                return math.atan2(y - ly, x - lx) if math.hypot(lx - x, ly - y) > 0.5 else math.atan2(vy, vx) + math.pi / 2
        return None

    def fight_move(self, me, t, target, sec_kind):
        x, y = me["pos"][0], me["pos"][1]
        tx, ty = target["pos"][0], target["pos"][1]
        d = target["_d"]
        lo, hi = BAND.get(sec_kind or "gun", BAND["gun"])
        if sec_kind is None:
            lo, hi = BAND["gun"]
        to = math.atan2(ty - y, tx - x)
        if d > hi:
            radial = 0.8
        elif d < lo:
            radial = -0.8
        else:
            radial = 0.0
        # constant-direction strafe; flip only when the side is blocked
        side = to + self.strafe * math.pi / 2
        if self.world.ray_clear(x, y, side, 6) < 5.5:
            self.strafe = -self.strafe
            side = to + self.strafe * math.pi / 2
        mx = math.cos(side) + radial * math.cos(to)
        my = math.sin(side) + radial * math.sin(to)
        return self.steer(me, math.atan2(my, mx), 5)

    def loot_goal(self, me, obs, visible, hpf, sec_kind, t):
        x, y = me["pos"][0], me["pos"][1]
        zone = obs["zone"]
        nc, nr = zone["next_center"], zone["next_radius"]
        safe = lambda p: math.hypot(p[0] - nc[0], p[1] - nc[1]) < nr + 40
        near_enemy = any(m["_d"] < 80 for m in visible)
        best, bs = None, 1e9
        # cubes: always worth a short walk, capped at the stat cap unless they are our kill drops
        if me.get("cubes", 0) < self.cube_cap + 4:
            for c in obs.get("cubes") or []:
                dd = d2(c["pos"], (x, y))
                if dd < 90 and safe(c["pos"]) and dd < bs:
                    best, bs = c["pos"], dd
        # recent kill spots (cubes may be out of view)
        self.kill_spots = [k for k in self.kill_spots if t - k[0] < 25]
        for k in self.kill_spots:
            dd = d2((k[1], k[2]), (x, y))
            if dd > 6 and dd < 110 and dd < bs:
                best, bs = (k[1], k[2]), dd
            elif dd <= 6:
                self.kill_spots.remove(k)
                break
        # weapons we want
        cur = SECONDARY_PREF.get(sec_kind, 0) if sec_kind else 0
        for p in obs.get("pickups") or []:
            if SECONDARY_PREF.get(p["kind"], -2) > max(cur, 0):
                dd = d2(p["pos"], (x, y)) * (0.5 if p["kind"] == "laser" else 0.8)
                if dd < 160 and safe(p["pos"]) and abs(p["pos"][2] - me["pos"][2]) < 3 and dd < bs:
                    best, bs = p["pos"], dd
        # supply drops: full repair; worth it when hurt, or close and quiet
        for dr in obs.get("drops") or []:
            if dr.get("lands_in", 99) > 6:
                continue
            dd = d2(dr["pos"], (x, y))
            if (hpf < 0.75 and dd < 180) or (dd < 90 and not near_enemy):
                if safe(dr["pos"]) and dd * 0.7 < bs:
                    best, bs = dr["pos"], dd * 0.7
        # crates for a weapon when we have no secondary
        if best is None and sec_kind is None and not near_enemy and hpf > 0.9:
            for c in obs.get("crates") or []:
                dd = d2(c["pos"], (x, y))
                if dd < 70 and safe(c["pos"]) and dd < bs:
                    best, bs = c["pos"], dd
        return best

    def aim_and_fire(self, act, me, t, target, visible, npcs, obs, sec, retreat, mode):
        x, y, z = me["pos"]
        cur = me["aim"]
        gun = (me.get("weapons") or [None])[0]
        sec_kind = sec["kind"] if sec else None
        aim = None
        fire0 = fire1 = False
        # dealing damage resets our regen timer, but always shoot back at a mech in our face
        holdfire = retreat and not (target is not None and target["_d"] < 60)
        if target is not None:
            tv = target.get("vel") or [0, 0, 0]
            gun_speed = (gun or {}).get("stats", {}).get("speed", self.w["gun"]["speed"]) if gun else 252
            aim = self.aim_at(me, target["pos"], tv, gun_speed, lead=P["lead"])
            d = target["_d"]
            if sec_kind == "laser" and sec.get("charge", 0) > 0.35:
                aim = self.aim_at(me, target["pos"], tv)  # hitscan: no lead while the beam is about to go
            if sec_kind == "mortar":
                pass
            yaw_err = abs(wrap(cur[0] - aim[0]))
            pit_err = abs(cur[1] - aim[1])
            tol = math.atan2(2.4, max(d, 1)) + 0.01
            on = yaw_err < tol and pit_err < tol * 1.5
            if not holdfire:
                if gun and gun.get("kind") == "gun" and d < (gun.get("stats", {}).get("range", 230)) and on:
                    fire0 = (gun.get("reload", 0) or 0) <= 0
                if sec_kind == "laser":
                    st = sec.get("stats", {})
                    lr = st.get("range", 280)
                    ch = sec.get("charge", 0) or 0
                    charging = ch > 0
                    if d < lr and (yaw_err < 0.25 or charging) and sec.get("cooldown", 0) <= 0.05:
                        fire1 = True
                    # release early (fires on release) when the partial charge is already lethal and we are on target
                    full, mn = st.get("charge", 0.75), st.get("min_charge", 0.15)
                    if charging and ch >= mn and on:
                        frac = st.get("min_frac", 0.35) + (1 - st.get("min_frac", 0.35)) * min(1.0, (ch - mn) / max(full - mn, 1e-3))
                        dmg = st.get("damage", 315) * frac * (1 + self.cube_bonus(me))
                        if dmg >= target.get("hp", 1400) + 5:
                            fire1 = False
                    if charging and not (target["_los"]) and ch >= mn:
                        fire1 = True  # keep holding; never dump a charge into a wall
                elif sec_kind == "lightning":
                    st = sec.get("stats", {})
                    cone = math.radians(st.get("cone_deg", 30)) / 2
                    if d < st.get("range", 45) * 0.95 and yaw_err < cone and me["energy"] > st.get("energy", 3) + 2:
                        fire1 = True
        elif mode == "loot":
            # crack a crate we are standing near (weapon chance), never while hurt
            crates = sorted(obs.get("crates") or [], key=lambda c: d2(c["pos"], me["pos"]))
            if crates and d2(crates[0]["pos"], me["pos"]) < 35 and me["hp"] > 0.9 * me["max_hp"]:
                c = crates[0]
                aim = self.aim_at(me, [c["pos"][0], c["pos"][1], c["pos"][2] - self.height * 0.5 + 1.2], [0, 0, 0])
                fire0 = abs(wrap(cur[0] - aim[0])) < 0.05 and abs(cur[1] - aim[1]) < 0.08
        # NPCs that are chewing on us
        if aim is None and not holdfire:
            close = [n for n in npcs if d2(n["pos"], me["pos"]) < 40 and self.los_to(me, n["pos"])]
            if close:
                n = min(close, key=lambda n: d2(n["pos"], me["pos"]))
                aim = self.aim_at(me, [n["pos"][0], n["pos"][1], n["pos"][2] - 2.5], n.get("vel") or [0, 0, 0], 252)
                fire0 = abs(wrap(cur[0] - aim[0])) < 0.06 and abs(cur[1] - aim[1]) < 0.1
        if aim is None:
            # face the most likely threat: recent hit, then nearest known mech, then a sound, then where we walk
            if self.hit_bearing and t - self.hit_bearing[0] < 3:
                aim = [self.hit_bearing[1], 0.0]
            elif visible:
                m = min(visible, key=lambda m: m["_d"])
                aim = self.aim_at(me, m["pos"], [0, 0, 0])
            else:
                snd = [s for s in obs.get("sounds") or [] if s.get("kind") in ("gun", "laser", "laser_charge", "step", "boost", "lightning", "reload")]
                if snd:
                    s = min(snd, key=lambda s: s.get("age", 0))
                    aim = [s["bearing"], 0.0]
                else:
                    mv = act["move"]
                    if math.hypot(*mv) > 0.1:
                        aim = [math.atan2(mv[1], mv[0]), 0.0]
                    else:
                        self.scan_yaw = wrap(self.scan_yaw + 0.25)
                        aim = [self.scan_yaw, 0.0]
            # keep laser charge only while a target is up
        if target is None and sec_kind == "laser" and (sec.get("charge", 0) or 0) > 0 and not holdfire:
            fire1 = True  # target ducked: keep the charge up rather than dumping it
        act["aim"] = [wrap(aim[0]), max(-0.6, min(1.2, aim[1]))]
        act["fire"] = [bool(fire0), bool(fire1)]
        if gun and gun.get("ammo", 12) < gun.get("stats", {}).get("mag", 12) * 0.5 and target is None and (gun.get("reload", 0) or 0) <= 0:
            act["reload"] = True

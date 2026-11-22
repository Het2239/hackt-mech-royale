"""Decision core. One `act` per observation; every branch returns a complete action.

Practice is us against a 15-bot house squad (shared vision, never fight each other, hunt us weakest-first),
and score = mechs that die before us, so kills dominate. The policy, in priority order:
  1. zone (arrival-time check, one goal per circle; final circle: stand on the centre)
  2. mortar dodge, lightning kite
  3. lose rule: short disengage to a spot hidden from the nearest threats, still shooting the target
  4. fight: engagement gate + target score, candidate-heading movement scored for range/LOS/exposure
  5. hunt a mech we were trading with after it breaks line of sight
  6. loot by value/(distance+18): laser first, cubes, crates; otherwise rotate inside the next circle
Phases: opening (quiet, laser-seeking, fight only what forces it), squad, cautious late game, FFA.
"""
import json
import math
import os
import random
import time

TAU = 2 * math.pi

P = {
    "open_secs": 24.0,         # opening phase length
    "lose_near2_hp": 0.55,     # disengage with 2+ mechs near below this HP fraction
    "lose_near3_hp": 0.80,     # ... with 3+ mechs near
    "lose_solo_hp": 0.35,      # disengage below this when the target is healthier than us
    "lose_hold_s": 1.5,        # keep disengaging this long after the last losing tick
    "engage_d": 60.0,          # fight anything this close
    "pref_gun": 85.0, "pref_laser": 120.0, "pref_lightning": 28.0,
    "finish_hp": 0.38,         # dive when the target is below this and we are healthy
    "dive_d": 20.0,
    "hunt_hp": 0.6, "hunt_s": 6.0,
    "lethal_k": 0.70,          # release the laser when 0.70 x partial damage kills
    "latency": 0.075,
    "cautious_t": 265.0,
    "w_los": 45.0, "w_exposed": 30.0, "w_strafe": 9.0,
    "v_weapon": 130.0, "v_laser": 320.0, "laser_seek": 240.0, "v_crate_noweap": 40.0,
    "scan": 1.0,
    "no_cover_fight": 1,
    "trust_vis": 0,
    "quiet": 1,                # walk quietly in the opening
    "grab_d": 30.0,            # mid-fight, detour to a better weapon this close
               # sweep the torso while looking for a weapon
}
try:
    P.update(json.loads(os.environ.get("BOT_PARAMS", "{}")))
except Exception:
    pass

LOG = open(os.environ["BOT_LOG"], "a") if os.environ.get("BOT_LOG") else None

SECONDARY_PREF = {"laser": 3, "lightning": 2, "mortar": -1}  # never carry the mortar: cookoff + loud
UPGRADE_ORDER = [
    "plating", "armor", "laser_damage", "laser_charge", "nanites", "gun_damage", "gun_rate",
    "lightning_efficiency", "lightning_range", "servos", "gun_mag", "gun_velocity", "gyros",
    "lightning_chain", "reactor", "capacitor", "laser_pierce", "sensors", "jets",
]


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def d2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(yaw):
    return [math.cos(yaw), math.sin(yaw)]


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class Brain:
    def __init__(self, hello, world):
        cfg = hello["config"]
        self.cfg = cfg
        self.world = world
        self.rng = random.Random(hello.get("seed", 0))
        m = cfg["mech"]
        self.walk = m.get("walk_speed", 11.7)
        self.eye = m.get("eye_height", 8.0)
        self.muzzle = m.get("muzzle_height", 7.0)
        self.height = m.get("height", 9.0)
        self.radius = m.get("radius", 2.5)
        self.turn = m.get("turn_rate", 4.0)
        self.pitch_rate = m.get("pitch_rate", 3.0)
        self.decide_dt = cfg["match"].get("decide_every", 2) / cfg["match"].get("tick_hz", 20)
        self.reach = cfg["loot"]["pickup_radius"]
        self.cube_cap = cfg["cubes"].get("stat_cap", 10)
        self.size_per_cube = cfg["cubes"].get("size_per_cube", 0.05)
        self.w = cfg["weapons"]
        self.light_r = self.w["lightning"].get("range", 45)
        snd = cfg.get("sound", {})
        self.quiet = clamp((snd.get("step_speed", 5) - 0.4) / self.walk, 0.2, 1.0)
        # memory
        self.attackers = {}      # id -> last time they damaged us
        self.hit_bearing = None  # (t, yaw)
        self.my_targets = {}     # id -> last time we damaged them
        self.kills = 0
        self.kill_spots = []
        self.last_seen = {}      # id -> (t, pos, hp, max_hp)
        self.dead_ids = set()
        self.alive0 = None
        self.ffa = False
        self.lose_until = -9.0
        self.cover = None
        self.cover_t = -9.0
        self.strafe = 1.0
        self.strafe_t = 0.0
        self.path = None
        self.path_goal = None
        self.path_t = -9.0
        self.stuck_t = 0.0
        self.unstuck_until = 0.0
        self.unstuck_yaw = 0.0
        self.last_pos = None
        self.zone_goal = None
        self.zone_key = None
        self.scan_yaw = 0.0
        self.target_id = None
        self.target_t = -9.0
        self.last_combat = -9.0
        self.loot_id = None
        self.loot_since = 0.0
        self.loot_ban = {}
        self.known_pickups = {}  # id -> (pos, kind, t_seen); remembered after we look away

    # ------------------------------------------------------------------ geometry helpers
    def eye_of(self, p):
        return (p[0], p[1], p[2] + self.eye)

    def body(self, p, f=0.5):
        return (p[0], p[1], p[2] + self.height * f)

    def can_see(self, a, b):
        return self.world.los(self.eye_of(a), self.body(b, 0.55)) or self.world.los(self.eye_of(a), self.body(b, 0.9))

    def zone_pen(self, p, obs):
        z = obs["zone"]
        t = obs["t"]
        pen = max(0.0, math.hypot(p[0] - z["center"][0], p[1] - z["center"][1]) - (z["radius"] - 6)) * 4
        if t > z["shrink_start"] - 15:
            pen += max(0.0, math.hypot(p[0] - z["next_center"][0], p[1] - z["next_center"][1]) - (z["next_radius"] - 4)) * 2
        return pen

    def steer(self, me, yaw, look=7.0):
        x, y = me["pos"][0], me["pos"][1]
        w = self.world
        if not w.ok or not w.walkable(x, y):
            return unit(yaw)
        for off in (0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05, 1.5, -1.5, 2.1, -2.1):
            if w.ray_clear(x, y, yaw + off, look) >= look - 0.1:
                return unit(yaw + off)
        return unit(yaw)

    def go(self, me, t, gx, gy, tol=3.0):
        x, y = me["pos"][0], me["pos"][1]
        if d2((x, y), (gx, gy)) < tol:
            return [0.0, 0.0]
        w = self.world
        if not w.ok:
            return self.steer(me, math.atan2(gy - y, gx - x))
        if (self.path is None or self.path_goal is None or d2(self.path_goal, (gx, gy)) > 8 or t - self.path_t > 2.0):
            self.path, self.path_goal, self.path_t = w.path(x, y, gx, gy), (gx, gy), t
        p = self.path
        if not p:
            return self.steer(me, math.atan2(gy - y, gx - x))
        while len(p) > 1 and d2(p[0], (x, y)) < 4.5:
            p.pop(0)
        tgt = p[0]
        for q in p[1:6]:
            dist = d2(q, (x, y))
            if w.ray_clear(x, y, math.atan2(q[1] - y, q[0] - x), dist) >= dist - 1.0:
                tgt = q
            else:
                break
        if len(p) == 1 and d2(p[0], (gx, gy)) < 6:
            tgt = (gx, gy)
        if d2(tgt, (x, y)) < 1.0:
            tgt = (gx, gy)
        return self.steer(me, math.atan2(tgt[1] - y, tgt[0] - x), look=4.0)

    # ------------------------------------------------------------------ aiming
    def aim_at(self, me, m, speed=None, frac=0.5):
        """[yaw, pitch] from our muzzle to the target body point, with latency + flight-time lead."""
        x, y, z = me["pos"]
        sv = me.get("vel") or [0, 0, 0]
        L = P["latency"]
        mx, my, mz = x + sv[0] * L, y + sv[1] * L, z + self.muzzle
        tp = m["pos"]
        tv = m.get("vel") or [0, 0, 0]
        hz = self.height * frac
        tf = L
        tx, ty, tz = tp[0], tp[1], tp[2] + hz
        for _ in range(4 if speed else 1):
            vz = tv[2] * (0.5 if abs(tv[2]) > 0.5 else 1.0)
            tx, ty, tz = tp[0] + tv[0] * tf, tp[1] + tv[1] * tf, tp[2] + hz + vz * tf
            if speed:
                tf = L + max(0.0, math.dist((mx, my, mz), (tx, ty, tz)) - 3) / speed
        dxy = math.hypot(tx - mx, ty - my)
        return [math.atan2(ty - my, tx - mx), math.atan2(tz - mz, max(dxy, 0.1))]

    def on_target(self, me, aim, m, d):
        r = self.radius * (1 + self.size_per_cube * m.get("cubes", 0))
        ye = abs(wrap(me["aim"][0] - aim[0])) - self.turn * self.decide_dt
        pe = abs(me["aim"][1] - aim[1]) - self.pitch_rate * self.decide_dt
        return ye < math.atan2(0.75 * r, d) and pe < math.atan2(0.4 * self.height, d)

    # ------------------------------------------------------------------ zone
    def zone_target(self, me, obs):
        z = obs["zone"]
        t = obs["t"]
        x, y = me["pos"][0], me["pos"][1]
        nc, nr = z["next_center"], z["next_radius"]
        key = (round(nc[0]), round(nc[1]), round(nr))
        if key != self.zone_key:
            self.zone_key = key
            dx, dy = x - nc[0], y - nc[1]
            d = math.hypot(dx, dy)
            inner = nr - clamp(0.3 * nr, 3, 30) - 5
            if nr <= 30:
                self.zone_goal = (nc[0], nc[1])
            elif d <= inner:
                self.zone_goal = (x, y)
            else:
                self.zone_goal = (nc[0] + dx / d * inner, nc[1] + dy / d * inner)
        dist_next = math.hypot(x - nc[0], y - nc[1]) - nr
        time_left = z["shrink_end"] - t
        need = max(0.0, dist_next) / self.walk
        cur_out = math.hypot(x - z["center"][0], y - z["center"][1]) > z["radius"] - 3
        urgent = cur_out or (dist_next > -5 and need + 10 > time_left)
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

        for e in obs.get("events") or []:
            k = e.get("type")
            if k == "damaged" and e.get("by") is not None:
                self.attackers[e["by"]] = t
                self.last_combat = t
                if e.get("bearing") is not None:
                    self.hit_bearing = (t, e["bearing"])
            elif k == "hit":
                self.my_targets[e["target"]] = t
                self.last_combat = t
            elif k == "kill":
                self.kills += 1
                self.dead_ids.add(e["target"])
                ls = self.last_seen.get(e["target"])
                if ls:
                    self.kill_spots.append((t, ls[1][0], ls[1][1]))
                self.attackers.pop(e["target"], None)

        mechs = [m for m in obs.get("mechs") or [] if m.get("kind") == "mech" and m["id"] not in self.dead_ids]
        npcs = [m for m in obs.get("mechs") or [] if m.get("kind") == "npc"]
        for m in mechs:
            self.last_seen[m["id"]] = (t, m["pos"], m.get("hp", 1400), m.get("max_hp", 1400))

        # FFA: deaths we did not cause (practice squads never kill each other)
        unexplained = self.alive0 - obs.get("alive", self.alive0) - self.kills
        if unexplained >= 3 or (unexplained >= 2 and t < 110) or (unexplained >= 1 and t < 35):
            self.ffa = True

        zone = obs["zone"]
        opening = t < P["open_secs"] and not self.ffa
        late = zone["next_radius"] <= 70 or obs.get("alive", 16) <= 3
        cautious = (t >= P["cautious_t"] or (zone["next_radius"] <= 72 and t >= 180)) and not self.ffa

        # upgrades
        off = obs.get("offer")
        if off and off.get("choices"):
            ids = [c["id"] for c in off["choices"]]
            act["pick"] = next((ids.index(u) for u in UPGRADE_ORDER if u in ids), 0)

        # weapons at our feet
        ws = me.get("weapons") or [None, None]
        gun = ws[0] if ws and ws[0] and ws[0].get("kind") == "gun" else None
        sec = ws[1] if len(ws) > 1 else None
        sec_kind = sec["kind"] if sec else None
        cur_pref = SECONDARY_PREF.get(sec_kind, 0) if sec_kind else 0
        for p in obs.get("pickups") or []:
            if d2(p["pos"], me["pos"]) < self.reach and SECONDARY_PREF.get(p["kind"], -2) > max(cur_pref, 0):
                act["take"] = p["id"]
                if sec is not None:
                    act["drop"] = 1
                break
        if sec_kind == "mortar" and "take" not in act:
            act["drop"] = 1

        seen_ids = set()
        for p in obs.get("pickups") or []:
            self.known_pickups[p["id"]] = (p["pos"], p["kind"], t)
            seen_ids.add(p["id"])
        for pid, (pp, pk, ts) in list(self.known_pickups.items()):
            if pid not in seen_ids and (d2(pp, me["pos"]) < 25 or t - ts > 90):
                del self.known_pickups[pid]  # should be visible from here but is not: gone

        # ---- perception
        for m in mechs:
            m["_d"] = d2(m["pos"], me["pos"])
            m["_los"] = m["_d"] < 260 and (P["trust_vis"] or self.can_see(me["pos"], m["pos"]))
            m["_hpf"] = m.get("hp", 1400) / max(m.get("max_hp", 1400), 1)
            dx, dy = x - m["pos"][0], y - m["pos"][1]
            dd = max(m["_d"], 0.1)
            v = m.get("vel") or [0, 0, 0]
            m["_closing"] = (v[0] * dx + v[1] * dy) / dd > 3
            m["_aiming"] = abs(wrap((m.get("aim") or [9, 0])[0] - math.atan2(dy, dx))) < 0.22
            m["_light"] = "lightning" in (m.get("weapons") or [])
        near = [m for m in mechs if m["_d"] < 160]

        # ---- engagement gate + target score
        target, best = None, 1e18
        for m in mechs:
            if not m["_los"] or m["_d"] > 200:
                continue
            d = m["_d"]
            hit_us = t - self.attackers.get(m["id"], -99)
            if self.ffa or late:
                ok = True
            elif opening:
                ok = (d < P["engage_d"] or hit_us < 4 or (m.get("charging") and m["_aiming"] and d < 120)
                      or (m.get("hp", 1400) < 350 and d < 120))
            else:
                ok = (d < P["engage_d"] or hit_us < 6 or (m["_closing"] and d < 180) or (m["_hpf"] < 0.4 and d < 160)
                      or (m["_aiming"] and d < 190))
            if not ok:
                continue
            s = d + 0.18 * m.get("hp", 1400)
            if m["_hpf"] <= 0.3:
                s -= 200
            if m["id"] == self.target_id and t - self.target_t < 2:
                s -= 200
            if hit_us < 4:
                s -= 40
            if m["_closing"]:
                s -= 15
            if s < best:
                best, target = s, m
        if target is not None:
            self.target_id, self.target_t = target["id"], t

        # ---- lose rule (practice only; never in the late game)
        losing = False
        if not self.ffa and not late:
            n = len(near)
            if (n >= 2 and hpf < P["lose_near2_hp"]) or (n >= 3 and hpf < P["lose_near3_hp"]):
                losing = True
            if target is not None and hpf < P["lose_solo_hp"] and target["_hpf"] > hpf + 0.10:
                losing = True
            if target is not None and target["_hpf"] < 0.25 and hpf > 0.20 and n <= 2:
                losing = False
            if target is not None and target["_d"] < 120 and (target.get("hp", 1400) < 200 or target.get("hp", 1400) < 0.5 * me["hp"]):
                losing = False  # finishing
        if losing:
            self.lose_until = t + P["lose_hold_s"]
        retreat = t < self.lose_until and bool(near)

        zgoal, urgent, outside = self.zone_target(me, obs)

        # ---- movement
        mode = "rotate"
        boost = False
        shells = [p for p in obs.get("projectiles") or [] if p.get("kind") == "mortar"]
        dodge = self.mortar_dodge(me, shells)
        lights = sorted([m for m in mechs if m["_light"] and m["_d"] < self.light_r + 6], key=lambda m: m["_d"])
        anti_light = bool(lights) and (sec_kind != "lightning" or hpf < 0.55)
        final = zone["radius"] <= 30 and zone["next_radius"] <= 25
        if t < self.unstuck_until:
            mode = "unstuck"
            act["move"] = unit(self.unstuck_yaw)
            act["jump"] = True
        elif outside or urgent:
            mode = "zone"
            act["move"] = self.go(me, t, *zgoal)
            boost = True
        elif dodge is not None and me["energy"] > 15:
            mode = "dodge"
            act["move"] = self.steer(me, dodge, 5)
            boost = True
        elif anti_light:
            mode = "kite"
            act["move"] = self.kite_lightning(me, lights[0], mechs, obs)
            boost = lights[0]["_d"] < self.light_r + 1 and me["energy"] > 15
        elif retreat:
            mode = "cover"
            threats = sorted(near, key=lambda m: m["_d"])[:4]
            if self.cover is None or t - self.cover_t > 0.4:
                self.cover = self.hidden_spot(me, [m["pos"] for m in threats], obs)
                self.cover_t = t
            if self.cover:
                act["move"] = self.go(me, t, self.cover[0], self.cover[1], tol=1.8)
                boost = me["energy"] > 10
            elif target is not None and P["no_cover_fight"]:
                mode = "fight"  # nowhere to hide: running only exposes our back (x1.5)
                act["move"] = self.fight_move(me, t, target, mechs, obs, gun, sec_kind, hpf, len(near))
            else:
                act["move"] = self.steer(me, math.atan2(y - threats[0]["pos"][1], x - threats[0]["pos"][0]))
                boost = me["energy"] > 10
        elif target is not None and self.grab_spot(me, sec_kind) is not None:
            mode = "grab"
            g = self.grab_spot(me, sec_kind)
            act["move"] = self.go(me, t, g[0], g[1], tol=0.5)
        elif target is not None:
            mode = "fight"
            act["move"] = self.fight_move(me, t, target, mechs, obs, gun, sec_kind, hpf, len(near))
            if sec_kind == "lightning" and target["_d"] > 50 and me["energy"] > 55:
                boost = True
        else:
            hunt = None if (opening or cautious) else self.hunt_target(me, t, hpf, obs)
            goal = None if hunt else self.loot_goal(me, obs, mechs, hpf, sec_kind, t, opening)
            if hunt is not None:
                mode = "hunt"
                act["move"] = self.go(me, t, hunt[0], hunt[1])
            elif goal is not None:
                mode = "loot"
                act["move"] = self.go(me, t, goal[0], goal[1], tol=1.0)
            elif final:
                mode = "final"
                c = zone["next_center"]
                act["move"] = self.go(me, t, c[0], c[1], tol=clamp(zone["next_radius"] * 0.06, 0.35, 2.0))
                boost = zone["next_radius"] < 18
            else:
                act["move"] = self.go(me, t, *zgoal)
            if P["quiet"] and opening and mode in ("rotate", "loot") and not boost:
                act["move"] = [act["move"][0] * self.quiet, act["move"][1] * self.quiet]

        # stuck detection
        mv = act["move"]
        if self.last_pos is not None and math.hypot(*mv) > 0.3 and mode != "unstuck":
            if math.hypot(x - self.last_pos[0], y - self.last_pos[1]) < 0.15:
                self.stuck_t += 0.1
            else:
                self.stuck_t = max(0.0, self.stuck_t - 0.05)
            if self.stuck_t > 1.2:
                self.stuck_t = 0.0
                self.unstuck_until = t + 1.0
                self.unstuck_yaw = math.atan2(mv[1], mv[0]) + self.rng.choice((-1.6, 1.6, math.pi))
                self.path = None
        self.last_pos = (x, y)
        act["boost"] = bool(boost and me["energy"] > 8)

        self.aim_and_fire(act, me, t, target, mechs, npcs, obs, gun, sec, mode, hpf)
        if LOG and int(t * 10) % 10 == 0:
            kp = sorted((d2(pp, me["pos"]), k) for pp, k, _ in self.known_pickups.values())[:2]
            LOG.write(f"{t:.0f} {mode} hp{hpf:.2f} sec={sec_kind} pk={[(round(a), b) for a, b in kp]} vis={len(obs.get('pickups') or [])} "
                      f"cr={len(obs.get('crates') or [])} tgt={target and round(target['_d'])} near={len(near)} take={act.get('take')} ev={[e.get('type') for e in obs.get('events') or []]}\n")
            LOG.flush()
        try:
            import mechfight as mf
            mf.draw.label(f"{mode} hp{hpf:.2f} k{self.kills}{' ffa' if self.ffa else ''}")
        except Exception:
            pass
        return act

    # ------------------------------------------------------------------ behaviours
    def grab_spot(self, me, sec_kind):
        cur = SECONDARY_PREF.get(sec_kind, 0) if sec_kind else 0
        best, bd = None, P["grab_d"]
        for pp, pk, _ts in self.known_pickups.values():
            if SECONDARY_PREF.get(pk, -2) > max(cur, 0) and abs(pp[2] - me["pos"][2]) < 3.2:
                d = d2(pp, me["pos"])
                if d < bd:
                    best, bd = pp, d
        return best

    def hunt_target(self, me, t, hpf, obs):
        if hpf < P["hunt_hp"]:
            return None
        best, bs = None, 1e18
        nc, nr = obs["zone"]["next_center"], obs["zone"]["next_radius"]
        for mid, (ts, pos, hp, mhp) in self.last_seen.items():
            if mid in self.dead_ids or t - ts > P["hunt_s"]:
                continue
            traded = t - self.my_targets.get(mid, -99) < P["hunt_s"] or t - self.attackers.get(mid, -99) < P["hunt_s"]
            if not traded or hp > me["hp"] * 1.1:
                continue
            if math.hypot(pos[0] - nc[0], pos[1] - nc[1]) > nr + 30:
                continue
            s = d2(pos, me["pos"]) + 0.1 * hp
            if s < bs:
                best, bs = pos, s
        return best

    def mortar_dodge(self, me, shells):
        x, y, z = me["pos"]
        g = self.w["mortar"].get("gravity", 22)
        r = self.w["mortar"].get("splash_radius", 7) + 4
        for p in shells:
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            a, b, c = -0.5 * g, vz, pz - z
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            tl = (-b - math.sqrt(disc)) / (2 * a)
            if tl <= 0 or tl > 3.5:
                continue
            lx, ly = px + vx * tl, py + vy * tl
            dd = math.hypot(lx - x, ly - y)
            if dd < r:
                return math.atan2(y - ly, x - lx) if dd > 0.5 else math.atan2(vy, vx) + math.pi / 2
        return None

    def kite_lightning(self, me, c, mechs, obs):
        x, y = me["pos"][0], me["pos"][1]
        away = math.atan2(y - c["pos"][1], x - c["pos"][0])
        best, bs = away, -1e18
        for rot in (0, 0.45, -0.45, 0.9, -0.9, 1.35, -1.35):
            a = away + rot
            px, py = x + 8 * math.cos(a), y + 8 * math.sin(a)
            if not self.world.walkable(px, py):
                continue
            s = 25 * math.cos(rot) - self.zone_pen((px, py), obs)
            for m in mechs:
                if m is c or m["_d"] > 170:
                    continue
                if self.world.los(self.eye_of(m["pos"]), (px, py, me["pos"][2] + self.height * 0.6)):
                    s -= 35 if m.get("charging") else 18
            if s > bs:
                best, bs = a, s
        return self.steer(me, best, 5)

    def hidden_spot(self, me, threats, obs):
        x, y, z = me["pos"]
        w = self.world
        best, bs = None, 1e18
        t_end = time.perf_counter() + 0.012
        for r in (6.0, 12.0, 20.0, 30.0, 42.0):
            if time.perf_counter() > t_end:
                break
            for k in range(16):
                a = k * TAU / 16
                px, py = x + r * math.cos(a), y + r * math.sin(a)
                if not w.walkable(px, py):
                    continue
                if w.ray_clear(x, y, a, r) < r - 2 and r > 12:
                    continue  # crude reachability
                pz = w.terrain(px, py) if w.ok else z
                seen = False
                for tp in threats:
                    e = self.eye_of(tp)
                    if w.los(e, (px, py, pz + self.height * 0.5)) or w.los(e, (px, py, pz + self.height * 0.9)):
                        seen = True
                        break
                if seen:
                    continue
                s = r + self.zone_pen((px, py), obs)
                if threats:
                    s -= 0.25 * min(min(d2((px, py), tp) for tp in threats), 80)
                if s < bs:
                    best, bs = (px, py), s
            if best is not None:
                break
        return best

    def pref_dist(self, me, target, gun, sec_kind, hpf, crowd):
        if sec_kind == "laser":
            pref = P["pref_laser"]
        elif sec_kind == "lightning" and me["energy"] > 12 and (target["_d"] < self.light_r + 25 or target["_hpf"] < 0.5):
            pref = P["pref_lightning"]
        else:
            pref = P["pref_gun"]
        tw = target.get("weapons") or []
        if sec_kind != "lightning" and "lightning" in tw:
            pref = max(pref, self.light_r + 19)
        if "mortar" in tw:
            pref = max(pref, 80)
        if target["_hpf"] < P["finish_hp"] and hpf > 0.42 and crowd <= 2:
            pref = P["dive_d"] + (4 if self.ffa else 0)
        return pref

    def fight_move(self, me, t, target, mechs, obs, gun, sec_kind, hpf, crowd):
        x, y, z = me["pos"]
        w = self.world
        tp = target["pos"]
        pref = self.pref_dist(me, target, gun, sec_kind, hpf, crowd)
        want = 1.0
        if sec_kind is None and gun is not None:
            if gun.get("ammo", 1) == 0 or (gun.get("reload", 0) or 0) > 0.25 or target["_d"] > gun.get("stats", {}).get("range", 230) - 10:
                want = -1.0  # gun-only and not ready: take cover from the target
        if t > self.strafe_t:
            self.strafe_t = t + 0.3 + 0.7 * self.rng.random()
            if self.rng.random() < 0.6:
                self.strafe = -self.strafe
        to = math.atan2(tp[1] - y, tp[0] - x)
        sdir = (math.cos(to + self.strafe * math.pi / 2), math.sin(to + self.strafe * math.pi / 2))
        others = sorted([m for m in mechs if m is not target and m["_d"] < 170], key=lambda m: m["_d"])[:4]
        cubes = [c["pos"] for c in obs.get("cubes") or [] if d2(c["pos"], me["pos"]) < 40]
        best, bs = None, -1e18
        cands = [(0.0, None)] + [(6.0, k * TAU / 16) for k in range(16)]
        for r, a in cands:
            if a is None:
                px, py = x, y
            else:
                px, py = x + r * math.cos(a), y + r * math.sin(a)
                if w.ok and (not w.walkable(px, py) or w.ray_clear(x, y, a, r) < r - 0.5):
                    continue
            pz = w.terrain(px, py) if w.ok else z
            if abs(pz - z) > 3:
                pz = z
            d = math.hypot(tp[0] - px, tp[1] - py)
            s = -1.1 * abs(d - pref) - 0.4 * r
            spot_eye = (px, py, pz + self.eye)
            if w.los(spot_eye, self.body(tp, 0.55)):
                s += P["w_los"] * want
            for m in others:
                if w.los(self.eye_of(m["pos"]), (px, py, pz + self.height * 0.6)):
                    s -= P["w_exposed"]
            for m in mechs:
                if m["_light"] and sec_kind != "lightning":
                    dl = math.hypot(m["pos"][0] - px, m["pos"][1] - py)
                    if dl < self.light_r + 16:
                        s -= (self.light_r + 16 - dl) * 4.2
            s -= self.zone_pen((px, py), obs)
            for c in cubes:
                s += 1.5 * (d2(c, (x, y)) - d2(c, (px, py)))
            if a is not None:
                if want > 0:
                    s += P["w_strafe"] * (math.cos(a) * sdir[0] + math.sin(a) * sdir[1])
                cl = w.ray_clear(px, py, a, 9) if w.ok else 9
                if cl < 3:
                    s -= 3 * (3 - cl)
            else:
                s -= 8
            if s > bs:
                best, bs = a, s
        if best is None:
            return [0.0, 0.0]
        return unit(best)

    def loot_goal(self, me, obs, mechs, hpf, sec_kind, t, opening):
        x, y = me["pos"][0], me["pos"][1]
        zone = obs["zone"]
        nc, nr = zone["next_center"], zone["next_radius"]
        danger = any(m["_d"] < 90 for m in mechs)
        cur = SECONDARY_PREF.get(sec_kind, 0) if sec_kind else 0
        cands = []
        for pid, (ppos, pkind, _ts) in self.known_pickups.items():
            pv = SECONDARY_PREF.get(pkind, -2)
            if pv > max(cur, 0):
                v = P["v_weapon"] if sec_kind is None else 60
                if pkind == "laser":
                    v = max(v, P["v_laser"])
                cands.append((("p", pid), ppos, v, P["laser_seek"] if pkind == "laser" else 160))
        ncubes = me.get("cubes", 0)
        for c in obs.get("cubes") or []:
            cands.append((("c", c["id"]), c["pos"], 75 if ncubes < self.cube_cap else 25, 105))
        self.kill_spots = [k for k in self.kill_spots if t - k[0] < 20 and d2((k[1], k[2]), (x, y)) > 5]
        for k in self.kill_spots:
            cands.append((("k", k[0]), (k[1], k[2], me["pos"][2]), 90, 110))
        if hpf > 0.5 and not danger:
            for c in obs.get("crates") or []:
                cands.append((("x", c["id"]), c["pos"], 0.6 * (75 if ncubes < self.cube_cap else 25) + (P["v_crate_noweap"] if sec_kind is None else 0), 90))
        best, bu = None, 0.0
        for key, pos, v, rad in cands:
            d = d2(pos, (x, y))
            if d > rad or t < self.loot_ban.get(key, -1):
                continue
            if len(pos) > 2 and abs(pos[2] - me["pos"][2]) > 3.2 and key[0] != "k":
                continue
            if math.hypot(pos[0] - nc[0], pos[1] - nc[1]) > nr + 5 and t > zone["shrink_start"] - 20:
                continue
            if self.world.ok and not self.world.walkable(pos[0], pos[1]):
                continue
            u = v / (d + 18)
            if key == self.loot_id:
                u *= 1.35
            if u > bu:
                best, bu = (key, pos), u
        if best is None:
            self.loot_id = None
            return None
        if best[0] != self.loot_id:
            self.loot_id, self.loot_since = best[0], t
        elif t - self.loot_since > 30:
            self.loot_ban[best[0]] = t + 60
            self.loot_id = None
            return None
        return best[1]

    def aim_and_fire(self, act, me, t, target, mechs, npcs, obs, gun, sec, mode, hpf):
        cur = me["aim"]
        sec_kind = sec["kind"] if sec else None
        aim = None
        fire0 = fire1 = False
        if target is not None:
            d = target["_d"]
            gspeed = (gun or {}).get("stats", {}).get("speed", self.w["gun"]["speed"])
            frac = 0.5 if self.world.los(self.eye_of(me["pos"]), self.body(target["pos"], 0.5)) else 0.85
            aim = self.aim_at(me, target, gspeed, frac)
            if sec_kind == "laser":
                st = sec.get("stats", {})
                full = st.get("charge", 0.75)
                ch = sec.get("charge", 0) or 0
                if ch > 0 and full - ch < 0.35:
                    aim = self.aim_at(me, target, None, frac)  # hitscan: no flight-time lead at release
            on = self.on_target(me, aim, target, d)
            if gun and d < gun.get("stats", {}).get("range", 230) and on and (gun.get("reload", 0) or 0) <= 0:
                fire0 = True
            if sec_kind == "laser":
                st = sec.get("stats", {})
                full, mn = st.get("charge", 0.75), st.get("min_charge", 0.15)
                ch = sec.get("charge", 0) or 0
                if d < st.get("range", 280) * 0.97 and sec.get("cooldown", 0) <= 0.05:
                    fire1 = True
                if ch >= mn and on:
                    mf = st.get("min_frac", 0.35)
                    dmg = st.get("damage", 315) * (mf + (1 - mf) * min(1.0, ch / full)) * (1 + self.cube_bonus(me))
                    if "mortar" in (target.get("weapons") or []):
                        dmg *= 1 + self.w["laser"].get("cookoff", 0.6)
                    if abs(wrap(target.get("aim", [0, 0])[0] - math.atan2(me["pos"][1] - target["pos"][1], me["pos"][0] - target["pos"][0]))) > math.pi / 2:
                        dmg *= self.cfg["mech"].get("rear_mult", 1.5)
                    if dmg * P["lethal_k"] >= target.get("hp", 1400):
                        fire1 = False  # release now: partial charge kills
                # the laser fires on release; held at full charge it only fires on its own after hold_s
                if ch >= full - 1e-6 and (on or ch >= full + st.get("hold_s", 2) - 0.3):
                    fire1 = False
            elif sec_kind == "lightning":
                st = sec.get("stats", {})
                cone = math.radians(st.get("cone_deg", 30)) / 2
                if d < st.get("range", 45) * 0.95 and abs(wrap(cur[0] - aim[0])) < cone and me["energy"] > st.get("energy", 3) + 2:
                    fire1 = True
        else:
            # pre-charge the laser while an engaged foe is out of sight
            if sec_kind == "laser" and self.target_id is not None and t - self.target_t < 3:
                st = sec.get("stats", {})
                ch = sec.get("charge", 0) or 0
                if ch < st.get("charge", 0.75) + st.get("hold_s", 2) - 0.35:
                    fire1 = True
            if mode == "loot" and hpf > 0.5:
                crates = sorted(obs.get("crates") or [], key=lambda c: d2(c["pos"], me["pos"]))
                if crates and d2(crates[0]["pos"], me["pos"]) < 35:
                    c = crates[0]
                    cm = {"pos": [c["pos"][0], c["pos"][1], c["pos"][2]], "vel": [0, 0, 0]}
                    aim = self.aim_at(me, cm, None, 1.2 / self.height)
                    fire0 = abs(wrap(cur[0] - aim[0])) < 0.05 and abs(cur[1] - aim[1]) < 0.08
            if aim is None:
                close = [n for n in npcs if d2(n["pos"], me["pos"]) < 45 and self.can_see(me["pos"], n["pos"])]
                if close:
                    n = min(close, key=lambda n: d2(n["pos"], me["pos"]))
                    aim = self.aim_at(me, n, self.w["gun"]["speed"], 0.2)
                    fire0 = abs(wrap(cur[0] - aim[0])) < 0.06 and abs(cur[1] - aim[1]) < 0.1
        if aim is None:
            if self.hit_bearing and t - self.hit_bearing[0] < 3:
                aim = [self.hit_bearing[1], 0.0]
            elif mechs:
                m = min(mechs, key=lambda m: m["_d"])
                aim = self.aim_at(me, m, None, 0.5)
            else:
                snd = [s for s in obs.get("sounds") or [] if s.get("kind") not in ("npc_gun", "crate", "drop")]
                if snd:
                    aim = [min(snd, key=lambda s: s.get("age", 0))["bearing"], 0.0]
                elif P["scan"] and (sec_kind != "laser") and mode in ("rotate", "loot"):
                    self.scan_yaw = wrap(cur[0] + 0.4)
                    aim = [self.scan_yaw, 0.0]
                elif math.hypot(*act["move"]) > 0.1:
                    aim = [math.atan2(act["move"][1], act["move"][0]), 0.0]
                else:
                    self.scan_yaw = wrap(self.scan_yaw + 0.3)
                    aim = [self.scan_yaw, 0.0]
        act["aim"] = [wrap(aim[0]), clamp(aim[1], -0.65, 1.25)]
        act["fire"] = [bool(fire0), bool(fire1)]
        if gun and target is None and (gun.get("reload", 0) or 0) <= 0:
            mag = gun.get("stats", {}).get("mag", 12)
            ammo = gun.get("ammo", mag)
            if ammo <= 0.6 * mag or (ammo < mag and t - self.last_combat > 3):
                act["reload"] = True

    def cube_bonus(self, me):
        c = self.cfg["cubes"]
        return min(me.get("cubes", 0), c.get("stat_cap", 10)) * c.get("damage_per_cube", 0.04)

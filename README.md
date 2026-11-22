# hackt — Mech Royale bot (`mechfight/1`)

Builders Buildathon 2026 entry, **Mech Royale** track. Python 3.13, standard library only.

## Layout

| Path | What |
| --- | --- |
| `bot.py` | Entrypoint; wraps the brain in a crash-safe `act` (a crash coasts on the last move). |
| `brain.py` | Decision core: zone, mortar dodge, anti-lightning, retreat-to-hide, selective fights, looting, upgrades, aim/fire. |
| `world.py` | Map model from `/arena/v2`: terrain, 3 m solid-top grid for line of sight, bounded A* navigation. |
| `mechfight.py`, `mechfight.schema.json`, `check/`, `arena/`, `tools/example_bot.py` | Official mechfight bot kit, unchanged. Only `mechfight.py` goes in the image. |

## Policy

Practice is us against 15 house bots that share vision and never fight each other, and
`score = 16 − place` = number of mechs that die before us. So:

1. Stay in the zone (arrival-time check against the shrink clock, one goal per circle).
2. Sidestep incoming mortar shells; keep out of lightning range of lightning carriers.
3. Below ~38 % HP, break line of sight from known threats and hold fire until regen gets us to ~85 %.
4. Fight selectively: mechs that attacked us, are close, or are nearly dead; constant-direction strafe at a
   range band per weapon; lead the gun; laser charge held on target; lightning inside its cone.
5. Snowball: scoop cubes (and our kill drops), grab laser > lightning (never the mortar), capture supply drops when hurt.
6. Switch to free-for-all targeting after 5 deaths we did not cause.

## Build and verify

```sh
docker build --platform linux/amd64 -t hackt-bot .
python3 check/check.py -- python3 -u bot.py
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none \
  --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m \
  --user 65534:65534 -v "$PWD/arena:/arena:ro" hackt-bot
```

## Attribution

- Official kit files listed above are from the mechfight bot kit on the track page
  (https://www.joinbuilders.org/buildathon/tracks/builders), unchanged.
- All bot code is our own. Strategy ideas (not code) were informed by public write-ups from other entrants:
  `ahmad-ovh/buildathon` (findings on scoring, dual-wield damage, anti-lightning, kill-drop scooping),
  `anthonyteti/buildathon-mech-royale` (selective engagement, cover while reloading),
  `MarwanMoh11/mech-royale` and `azz0048/Mech-Royale` (laser-first, no mortar, retreat/heal thresholds).
- Released under the MIT License (see `LICENSE`).

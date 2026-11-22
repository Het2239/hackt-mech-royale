# Mech Royale bot — Apex-Go v2 (`mechfight/1`)

This branch is an **unmodified copy** of `ahmad-ovh/buildathon` at commit
[`a484109`](https://github.com/ahmad-ovh/buildathon/commit/a484109) ("Apex-Go v2"),
by ahmad-ovh, released under the MIT License (see `LICENSE`, copyright ahmad-ovh).
Only the files needed to build the image plus the official kit are included; the original
documentation lives in the upstream repository.

Submitted by team hackt with the approval of the Builders Buildathon organizers.
Our own bot (Python, written from scratch) is on the `main` branch.

## Build and verify

```sh
docker build --platform linux/amd64 -t mech-bot .
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none \
  --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m \
  --user 65534:65534 -v "$PWD/arena:/arena:ro" mech-bot
```

Official kit files (`mechfight.py`, `mechfight.schema.json`, `check/`, `arena/`, `tools/example_bot.py`)
are from the mechfight bot kit on the track page, unchanged.

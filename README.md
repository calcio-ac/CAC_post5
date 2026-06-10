# CAC_post5 — France at World Cup 2022: Passing Networks

**Calcio AC analytics pipeline #05.** One passing network per France match at Qatar 2022 — seven games from the group stage to the final — built from StatsBomb open event data, served as an interactive site and exported as 1080×1350 images for Instagram / LinkedIn.

**Live site:** [calcio-ac.github.io/CAC_post5](https://calcio-ac.github.io/CAC_post5/) — all seven networks in France's national palette (dark blue `#21304D` / blue `#17548C` / red `#ED2939`), with a min-passes-per-line slider (default 2+), attempted-vs-completed pass totals, per-match HD download (2160×2700) and a download-all button. Networks are drawn on `<canvas>` straight from `web/data.json`, so downloads are pixel-perfect.

## Contents

| File | What it is |
|---|---|
| [`index.html`](index.html) | Interactive dashboard: match tabs, threshold slider, all-matches grid, HD downloads. No build step. |
| [`web/data.json`](web/data.json) | Nodes + edges + metadata for all 7 networks, exported from the notebook pipeline. |
| [`france_passing_networks.ipynb`](france_passing_networks.ipynb) | Full pipeline: fetch → network build → CAC-styled render. Executed with outputs. |
| [`videos/france_networks_4k.mp4`](videos/france_networks_4k.mp4) | 75 s cinematic film (16:9, 4K, 24 fps): the seven networks morphing match-to-match with possession / passes / xG callouts and an all-seven finale. Rendered by [`make_video.py`](make_video.py). |
| [`images/`](images/) | The 9-slide carousel: cover, 7 match networks (chronological), methodology. 2160×2700 px (2x HD). |

## Method

- **Data**: [StatsBomb open data](https://github.com/statsbomb/open-data) (`statsbombpy`), FIFA World Cup 2022, all 7 France matches.
- **The XI**: the 11 France players with the **most minutes in that match** (lineup position spans, extra time included, shootouts excluded) — full-game data, no substitution cutoff.
- **Edges**: completed passes between those 11 across the whole match (both directions summed); drawn from 4+ combinations on the slides, threshold adjustable on the site (default 2+).
- **Nodes**: position = average location of a player's passes *and* receptions; size = total passes involved in.
- **Totals**: each card also reports all France passes for the match, attempted vs completed. These maps describe build-up structure, not formations.

## Reproduce

```bash
pip install statsbombpy pandas mplsoccer matplotlib
jupyter nbconvert --to notebook --execute --inplace france_passing_networks.ipynb
```

First run fetches events from StatsBomb (a few minutes) and caches them locally as `.pkl` (gitignored); re-runs are instant.

---

*Part of the [Calcio AC](https://calcioac.com) post series — one repo per post: travel fatigue ([CAC_post1](https://github.com/calcio-ac/CAC_post1)), Salah's role change ([CAC_post2](https://github.com/calcio-ac/CAC_post2)), where the World Cup lives ([CAC_post3](https://github.com/calcio-ac/CAC_post3)), Portugal squad report ([CAC_post4](https://github.com/calcio-ac/CAC_post4)).*

Data © StatsBomb — free open data used under their [terms](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).

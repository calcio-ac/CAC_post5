# CAC_post5 — France at World Cup 2022: Passing Networks

**Calcio AC analytics pipeline #05.** One passing network per France match at Qatar 2022 — seven games from the group stage to the final — built from StatsBomb open event data and exported as 1080×1350 images for Instagram / LinkedIn.

## Contents

| File | What it is |
|---|---|
| [`france_passing_networks.ipynb`](france_passing_networks.ipynb) | Full pipeline: fetch → network build → CAC-styled render. Executed with outputs. |
| [`images/`](images/) | The 9-slide carousel: cover, 7 match networks (chronological), methodology. 2160×2700 px (2x HD). |

## Method

- **Data**: [StatsBomb open data](https://github.com/statsbomb/open-data) (`statsbombpy`), FIFA World Cup 2022, all 7 France matches.
- **Window**: kickoff until France's **first substitution**, so the XI on the map actually played together. Completed passes only.
- **Nodes**: position = average location of a player's passes *and* receptions; size = total passes involved in.
- **Edges**: pass count between a pair (both directions summed), drawn from 4+ combinations; width and darkness scale with volume.
- **Caveats**: early subs shrink the window — vs Australia it closes at 12' (Lucas Hernández injury, 46 passes), in the final at 40' (the Giroud + Dembélé double sub, 116 passes). These maps describe build-up structure, not formations.

## Reproduce

```bash
pip install statsbombpy pandas mplsoccer matplotlib
jupyter nbconvert --to notebook --execute --inplace france_passing_networks.ipynb
```

First run fetches events from StatsBomb (a few minutes) and caches them locally as `.pkl` (gitignored); re-runs are instant.

---

*Part of the [Calcio AC](https://calcioac.com) post series — one repo per post: travel fatigue ([CAC_post1](https://github.com/calcio-ac/CAC_post1)), Salah's role change ([CAC_post2](https://github.com/calcio-ac/CAC_post2)), where the World Cup lives ([CAC_post3](https://github.com/calcio-ac/CAC_post3)), Portugal squad report ([CAC_post4](https://github.com/calcio-ac/CAC_post4)).*

Data © StatsBomb — free open data used under their [terms](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).

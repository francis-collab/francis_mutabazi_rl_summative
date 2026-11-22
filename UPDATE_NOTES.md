# Fix: agent no longer locked on plot 5

## Root cause
`action_mask()` set `MOVE_NEXT = False` once every plot had been inspected.
After the early tour the agent **could not leave plot 5** — training and play both respected the mask.

GitHub master had no such lock; that is why it kept moving.

## Fixes in this package
1. **action_mask** — `MOVE_NEXT` and `WAIT` stay available all season (except final harvest-only day).
2. **Stronger MOVE** when other plots are diseased (`0.70 + 2.2 * other_disease`).
3. **Anti-camping penalty** if the agent stays ≥2 steps on one plot while another plot has disease ≥ 0.12.
4. **Treatment coverage** bonus at harvest (`+10 × fraction of plots treated`).
5. Tests updated; all 6 env tests pass.

Observation/action spaces unchanged. Same env for DQN / REINFORCE / A2C / PPO.

## Apply
```bash
unzip -o env_mask_circulation_fix.zip
# optional all 40 (required — policies learned under the old illegal MOVE mask)
uv run run_experiments.py --device cuda --total-steps 80000
uv run main.py --mode play --model models/dqn/best_model/dqn_model.pth --episodes 1 --viz 2d
```
Expect movement between plots whenever disease is visible on non-current plots.

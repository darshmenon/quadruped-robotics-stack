# Local Go2 model files

```bash
python3 scripts/download_pretrained.py
python3 scripts/download_pretrained.py --only stairs
```

Docs: [docs/PRETRAINED.md](../../docs/PRETRAINED.md).

| Folder | Contents |
|--------|----------|
| `go2_locomotion/` | Flat + rough Isaac PPO |
| `go2_parkour/` | RPL / visual distill parkour |
| `go2_walk_trotprior/` | MuJoCo PPO + clocked trot prior walker (`go2_flat_walk.zip` + `vecnorm_final.pkl`); needs `GO2_TROT_PRIOR=1` |
| `go2_stairs/` | Genesis sim2real walk/stairs + CTS MoE stairs policy |

These `*.pt` files are not Stable-Baselines3 zips. For the MuJoCo stairs course, train the local SB3 model:

```bash
./scripts/train_stairs.sh --blind --init-from-flat --n_envs 4 --timesteps 300000 --device cpu
```

Headless SB3 checks live outside this folder:

```bash
python3 training/play_policy.py \
  --model training/logs/stairs/best_model.zip \
  --vecnorm training/logs/stairs/vecnorm_final.pkl \
  --scene stairs --blind \
  --no-display --episodes 2 --max-steps 300
```

# Leaderboard

## Public leaderboard

You can see the [public leaderboard on Kaggle](https://www.kaggle.com/c/cartesius/leaderboard).

## Private leaderboard

!!! info
    This leaderboard is just used to keep track of the models implemented in `cartesius` and how they compare to each other.

| Config name                   |                                                           Checkpoint                                                          | Total loss |    area   |  centroid  | concavity |  min_clear | perimeter | size       |
| ----------------------------- | :---------------------------------------------------------------------------------------------------------------------------: | :--------: | :-------: | :--------: | :-------: | :--------: | :-------: | ---------- |
| graph_base_static_scale.yaml  |  [*ckpt*](https://github.com/TeamSPWK/cartesius/releases/download/untagged-069a5fb3505f82ae6f83/graph_base_static_scale.ckpt) |  **5.455** | *0.00845* | *0.0007111* |  *0.2038* | *0.003739* |  *0.2713* | *0.003037* |
| transformer_lr_sched.yaml | [*ckpt*](https://github.com/TeamSPWK/cartesius/releases/download/untagged-20a90d4ca33d1136b502/transformer_lr_sched.ckpt) |  **5.628**  | *0.008551* | *0.001205* |  *0.213* | *0.005635* |  *0.1383* | *0.004007* |
| transformer_static_scale.yaml | [*ckpt*](https://github.com/TeamSPWK/cartesius/releases/download/untagged-069a5fb3505f82ae6f83/transformer_static_scale.ckpt) |  **6.204**  | *0.01208* | *0.001488* |  *0.2136* | *0.006439* |  *0.2888* | *0.004647* |
| transformer_avg.yaml | [*ckpt*](https://github.com/TeamSPWK/cartesius/releases/download/untagged-069a5fb3505f82ae6f83/transformer_avg.ckpt) |  **6.696**   | _0.01132_ | _0.001984_ |   _0.2026_  |  _0.004641_  |   _0.9786_  | _0.006836_ |
| transformer.yaml              |        [*ckpt*](https://github.com/TeamSPWK/cartesius/releases/download/untagged-069a5fb3505f82ae6f83/transformer.ckpt)       |  **8.441** | *0.02875* | *0.001259* |  *0.2078* |  *0.0342* |  *0.6036* | *0.001191* |
| graph_base.yaml               |        [*ckpt*](https://github.com/TeamSPWK/cartesius/releases/download/untagged-069a5fb3505f82ae6f83/graph_base.ckpt)        |  **8.719** | *0.0411* | *0.001698* |  *0.1362* |   *0.007201*  |  *1.346* | *0.006203* |
| se3.yaml | [*ckpt*](https://github.com/TeamSPWK/cartesius/releases/download/untagged-069a5fb3505f82ae6f83/se3.ckpt) |    **11.806**   | _0.03225_ | _0.012_ |   _0.2879_  |  _0.007499_  |   _0.2545_  | _0.03637_ |

!!! tip
    You can reproduce each result by downloading the checkpoint and running :
    ```bash
    cartesius train=False config=<config_name.yaml> ckpt=<path/to/local.ckpt>
    ```

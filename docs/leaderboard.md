# Leaderboard

## Public leaderboard

You can see the [public leaderboard on Kaggle](https://www.kaggle.com/c/cartesius/leaderboard).

## Private leaderboard

!!! info
    This leaderboard is just used to keep track of the models implemented in `cartesius` and how they compare to each other.

| Config name                   |                                                           Checkpoint                                                          | Test loss |
| ----------------------------- | :---------------------------------------------------------------------------------------------------------------------------: | :--------: |
| graph_base.yaml  |  [*ckpt*](https://github.com/TeamSPWK/cartesius/releases/download/untagged-d0f96c06e59f279bfe4a/graph_base.ckpt) |  **5.455** |
| transformer_lr_sched.yaml | [*ckpt*](https://github.com/TeamSPWK/cartesius/releases/download/untagged-d0f96c06e59f279bfe4a/transformer_lr_sched.ckpt) |  **5.628**  |
| transformer_max.yaml | [*ckpt*](https://github.com/TeamSPWK/cartesius/releases/download/untagged-d0f96c06e59f279bfe4a/transformer_max.ckpt) |  **5.882**  |
| transformer.yaml | [*ckpt*](https://github.com/TeamSPWK/cartesius/releases/download/untagged-d0f96c06e59f279bfe4a/transformer.ckpt) |  **6.204**  |
| transformer_avg.yaml | [*ckpt*](https://github.com/TeamSPWK/cartesius/releases/download/untagged-d0f96c06e59f279bfe4a/transformer_avg.ckpt) |  **6.696**   |
| se3.yaml | [*ckpt*](https://github.com/TeamSPWK/cartesius/releases/download/untagged-d0f96c06e59f279bfe4a/se3.ckpt) |    **11.806**   |

!!! tip
    You can reproduce each result by downloading the checkpoint and running :
    ```bash
    cartesius train=False config=<config_name.yaml> ckpt=<path/to/local.ckpt>
    ```

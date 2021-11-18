# Leaderboard

| Config name                   |                                                           Checkpoint                                                          | Total loss |    area   |  centroid  | concavity |  min_clear | perimeter | size       |
| ----------------------------- | :---------------------------------------------------------------------------------------------------------------------------: | :--------: | :-------: | :--------: | :-------: | :--------: | :-------: | ---------- |
| transformer_static_scale.yaml | [*ckpt*](https://github.com/TeamSPWK/cartesius/releases/download/untagged-79e7436c9dd96892b8d4/transformer_static_scale.ckpt) |  **6.54**  | *0.01312* | *0.001943* |  *0.2123* | *0.006937* |  *0.3726* | *0.007476* |
| graph_base_static_scale.yaml  |  [*ckpt*](https://github.com/TeamSPWK/cartesius/releases/download/untagged-79e7436c9dd96892b8d4/graph_base_static_scale.ckpt) |  **6.557** | *0.01596* | *0.001235* |  *0.2101* | *0.002985* |  *0.2356* | *0.008056* |
| graph_base.yaml               |        [*ckpt*](https://github.com/TeamSPWK/cartesius/releases/download/untagged-79e7436c9dd96892b8d4/graph_base.ckpt)        |  **6.719** | *0.03215* | *0.001509* |  *0.1303* |   *0.013*  |  *0.5147* | *0.000932* |
| transformer.yaml              |        [*ckpt*](https://github.com/TeamSPWK/cartesius/releases/download/untagged-79e7436c9dd96892b8d4/transformer.ckpt)       |  **7.609** | *0.02422* | *0.001456* |  *0.2056* |  *0.01153* |  *0.6108* | *0.003209* |

!!! tip
    You can reproduce each result by downloading the checkpoint and running :
    ```bash
    cartesius train=False config=<config_name.yaml> ckpt=<path/to/local.ckpt>
    ```

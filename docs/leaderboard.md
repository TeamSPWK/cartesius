# Leaderboard

| Config file                   | Total loss |   area  | centroid | concavity | min_clear | perimeter |   size   |
|-------------------------------|:----------:|:-------:|:--------:|:---------:|:---------:|:---------:|:--------:|
| transformer_static_scale.yaml |    **6.54**    | _0.01312_ | _0.001943_ |   _0.2123_  |  _0.006937_ |   _0.3726_  | _0.007476_ |
| graph_base_static_scale.yaml     |    **6.827**   | _0.01502_ | _0.002365_ |   _0.2186_  |  _0.005064_  |   _0.1718_  | _0.01125_ |
| graph_base.yaml              |    **7.584**   | _0.03185_ | _0.001749_ |   _0.1647_  |  _0.01072_  |   _0.6529_  | _0.003004_ |
| transformer.yaml              |    **7.609**   | _0.02422_ | _0.001456_ |   _0.2056_  |  _0.01153_  |   _0.6108_  | _0.003209_ |

!!! tip
    You can reproduce each result by downloading the checkpoint and running :
    ```bash
    cartesius train=False config=<config_name.yaml> ckpt=<path/to/local.ckpt>
    ```

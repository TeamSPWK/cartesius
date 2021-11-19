# Downstream tasks

The goal of `cartesius` is to have a **benchmark** for **feature extraction** from **polygon data**.

For this, we train the models on **several tasks** related to **polygon representation**, which _may_ help the models to learn **proper features**.

If the model is learning proper features, then we can also think of `cartesius` as a **pretraining** framework, and use the pretrained models in **downstream tasks** (and hopefully it will help the agent to learn faster/better on the downstream task).

!!! warning "Disclaimer"
    `cartesius` is still under heavy development. It's absolutely possible that the current set of tasks are nowhere near enough to enable the model to learn good features. Maybe the pretraining is useless for downstream tasks. For now, we can't say for sure.

## Usage

### Instanciating the model

You can simply import and instanciate the right class from `cartesius`.  
For example for a Transformer model :

```python
# Import the model from cartesius
from cartesius.models import Transformer

# Instanciate your model with the parameters of your choice
m = Transformer(d_model=32, max_seq_len=64, n_heads=8, d_ff=64, dropout=0, activation="gelu", n_layers=2)
```

### Tokenizing polygons

Each model has a specific way to **tokenize polygons**. You can just import the right tokenizer, and use it to tokenize several polygons into a batch. It will return a `dict` of `torch.Tensor`.  
For example for a Transformer model, we use the `TransformerTokenizer`:

```python
from shapely.geometry import Polygon

# Import the tokenizer
from cartesius.tokenizers import TransformerTokenizer

# Instanciate the tokenizer with the parameters of your choice
tokenizer = TransformerTokenizer(max_seq_len=256)

polys = [
    Polygon([(0, 0), (0, 1), (1, 0), (0, 0)]),
    Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
]
# Batch together a list of polygons. `batch` is a dictionary of tensors
batch = tokenizer(polys)
```

!!! warning
    In `cartesius`, polygons are usually normalized. It is **your** responsability to normalize polygons properly.  
    You may use the normalization functions of `cartesius` : check out [Transforms](../code_ref/data#transforms)

### Call your model

You can then call your model with the tokenized data :

```python
features = m(polygon=batch["polygon"], mask=batch["mask"])

# Or simply :
features = m(**batch)
```

!!! info
    The model is a `nn.Module`, so you can use it as any Pytorch module !

## Pretrained models

You can see a list of the trained models and their score on the [Leaderboard](leaderboard.md).

### Loading a pretrained checkpoint

After downloading the checkpoint you're interested in, you can just load it in your model with `load_ckpt_state_dict()` :

```python
from cartesius.models import Transformer
from cartesius.utils import load_ckpt_state_dict

# Instanciate your model with the same parameters as the checkpoint
m = Transformer(d_model=32, max_seq_len=64, n_heads=8, d_ff=64, dropout=0, activation="gelu", n_layers=2)

# Load the state_dict from the checkpoint
m.load_state_dict(load_ckpt_state_dict("path/to/downloaded.ckpt"))
```

!!! info
    If needed, you can specify another device when loading the checkpoint, by passing `map_location` argument :
    ```python
    load_ckpt_state_dict("path/to/downloaded.ckpt", map_location=torch.device("cpu"))
    ```

### Use specific configuration

You can load specific configuration file from `cartesius` easily with `load_yaml()` :

```python
from cartesius.models import Transformer
from cartesius.utils import load_yaml

# Get the configuration from `transformers.yaml`
conf = load_yaml("transformers.yaml")

# Instanciate your model following the config loaded
m = Transformer(d_model=conf.d_model, max_seq_len=conf.max_seq_len, n_heads=conf.n_heads, d_ff=conf.d_ff, dropout=conf.dropout, activation=conf.activation, n_layers=conf.n_layers)
```

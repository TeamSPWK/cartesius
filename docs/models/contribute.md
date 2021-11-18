# Create your own

The goal of `cartesius` is to share a **common benchmark / experiments** across projects, to find which architecture / parameters are the **best for representing geometries**.

If you think you have an architecture that can achieve better scores, please add it to `cartesius` and benchmark it !

## Create your own...

### Model

To add a new model, follow these steps :

* Implement the model in a new `py` file in the folder `cartesius/models`
* Update the `create_model()` function in `cartesius/models/__init__.py` to initialize your model with the right arguments (from configuration)

!!! question
    If you just want to update an existing model, go ahead and modify it, but the default configuration should have a consistent behavior accross versions !

### Tokenizer

Creating a new model is not the only way to achieve better results !

Input representation plays an important role in model's performances. If you want to make your own `Tokenizer`, follow these steps :

* Implement the tokenizer in the `cartesius/tokenizers.py` file
* Your tokenizer should be a subclass of the `Tokenizer` class, and overwrite the `tokenize()` method (and optionally `__init__()`)
* Update the `TOKENIZERS` dictionary (in the same file) to add your tokenizer

### Transform

Similarly, data normalization can be modified to achieve better results.

To add your custom data normalization, follow these steps :

* Implement your code in the `cartesius/transforms.py` file
* Your data normalization code should be a subclass of the `Transform` class, and overwrite the `__call__()` method.
* Update the `TRANSFORMS` dictionary (in the same file) to add your transform

### Task

Tasks are important to ensure the models are learning to extract proper features. It's also used to compare the models on the test set.

To create a new task, follow these steps :

* Implement the task in the `cartesius/tasks.py` file
* Your task should be a subclass of the `Task` class, and overwrite the `get_label()` method (and optionally `__init__()`, `get_head()`, `get_loss_fn()`)
* Update the `TASKS` dictionary (in the same file) to add your task

## Contribution

Once you added your implementation, you can create a new configuration file (in `cartesius/config` folder) with the best parameters for your architecture, and train your model with `cartesius` command.

It will train your model and evaluate it on the test set. You can now compare it with other model's results.

!!! note
    If your changes add a new key in the configuration, don't forget to add it in the `cartesius/config/default.yaml` configuration !

---

Before opening a PR with your new implementation, don't forget to :

* Update the documentation to explain your new feature !
* Add some unit-tests

!!! tip
    Unit-testing neural network is difficult, but at least ensure the forward pass does not crash, and gradients are not `None` when backpropagating.

See [Contribute](../index.md#contribute) for additional details.

## Leaderboard

After training your model with your configuration, don't forget to update the [Leaderboard](../leaderboard.md) with your results !

Add a row with the scores of your checkpoint (ordered by `total loss`), and put the name of the config file you used for training.

Also, you should upload your checkpoint and put the link in the leaderboard (for reproducibility).  
More details in the [Upload your checkpoint](./upload_ckpt.md) page.

# Create your own

The goal of `cartesius` is to share a **common benchmark / experiments** across projects, to find which architecture / parameters are the **best for representing geometries**.

If you think you have an architecture that can achieve better scores, please add it to `cartesius` and benchmark it !

## Create your own model

!!! faq
    If you just want to update an existing model, go ahead and modify it, but you should ensure the default configuration does not introduce changes ! 

To add a new model, follow these steps :

* Implement the model in a new `py` file in `cartesius/models`
* Update the `create_model()` function in `cartesius/models/__init__.py` to initialize your model with the right arguments (from configuration)
* Update the default configuration appropriately (in `cartesius/config/default.yaml`)

---

Once you added your implementation, you can create a new configuration file (in `cartesius/config` folder) with the best parameters for your architecture, and train your model with `cartesius` command.

It will train your model and evaluate it on the test set. You can now compare it with other model's results.

---

Before opening a PR with your new architecture, don't forget to :

* Add some documentation explaining your architecture
* Add unit-tests

!!! tip
    Unit-testing neural network is difficult, but at least ensure the forward pass does not crash, and gradients are not `None` when backpropagating.

## Create your own Tokenizer

Creating a new model is not the only way to achieve better results !

Input representation plays an important role in model's performances. If you want to make your own `Tokenizer`, follow these steps :

* Implement the tokenizer in the `cartesius/tokenizers.py` file
* Your tokenizer should be a subclass of the `Tokenizer` class, and overwrite the `tokenize()` method (and optionally `__init__()`)
* Update the `TOKENIZERS` dictionary (in the same file) to add your tokenizer

---

Ensure you also add some unit-tests ! Tokenizers are self-contained, it shouldn't be too hard to test.

If possible, also update the documentation.

## Create your own Task

Tasks are important to ensure the models are learning to extract proper features. It's also used to compare the models on the test set.

To create a new task, follow these steps :

* Implement the task in the `cartesius/tasks.py` file
* Your task should be a subclass of the `Task` class, and overwrite the `get_label()` method (and optionally `__init__()`, `get_head()`, `get_loss_fn()`)
* Update the `TASKS` dictionary (in the same file) to add your task

---

Make sure you update the documentation with your new task !

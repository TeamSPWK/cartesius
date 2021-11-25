# SE(3)-Transformer

## General

The _SE(3)-Transformer_ is a variant of the self-attention module, **translation-equivariant** and **rotation-equivariant**.

📝 [Paper](https://arxiv.org/pdf/2006.10503.pdf)

!!! info
    This model uses the implementation from [_lucidrains_](https://github.com/lucidrains/se3-transformer-pytorch).

## Tokenization

It uses the same tokenization scheme as the basic Transformer.  
See [Tokenization](../transformer#tokenization) for more information.

## Feature extraction

Tokens are fed into the _SE(3)-Transformer_, and the representation of the whole polygon is computed by **mean-pooling** the representation of each node.

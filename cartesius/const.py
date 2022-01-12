class Const:
    """Constant class
    """

    @classmethod
    def __init__(cls, d):
        for k, v in d.items():
            setattr(cls, k, cls.const(v))

    @staticmethod
    def raise_(e):
        raise e

    @classmethod
    def const(cls, v):
        const_err = TypeError("Constant cannot be dynamically assigned")
        return property(lambda self: v, lambda self, assign: cls.raise_(const_err))


const_dict = {
    "epsilon": 1e-9,
}

CONST = Const(const_dict)

import torch
import torch.nn as nn
import inspect
import functools

#用于将调用函数时传递的参数保存到类的 config 字典中

#装饰器函数的定义，接受一个函数 func 作为参数，这个函数通常是一个类的方法

#这个装饰器函数用于在类的方法调用时，将传递的参数和值保存到类的 config 字典中。
#这在模型加载时很有用，可以将模型的配置信息存储在对象中，以便在需要的时候使用。
def store_config_args(func):
    """
    Class-method decorator that saves every argument provided to the
    function as a dictionary in 'self.config'. This is used to assist
    model loading - see LoadableModel.
    """
    #数获取传入函数的参数信息，包括参数名称、可变参数、关键字参数和默认值。
    attrs, varargs, varkw, defaults = inspect.getargspec(func)
    #装饰器的内部装饰器，用于保留原始函数的元信息。
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        #首先创建一个空的 config 字典，用于存储参数和值。
        self.config = {}

        # first save the default values
        #处理默认值：如果原始函数有默认值，将这些默认值添加到 config 字典中。
        if defaults:
            for attr, val in zip(reversed(attrs), reversed(defaults)):
                self.config[attr] = val

        # next handle positional args
        #将传入的位置参数与参数名称一一对应，将参数名作为键，参数值作为值，存储到 config 字典中。
        for attr, val in zip(attrs[1:], args):
            self.config[attr] = val

        # lastly handle keyword args
        #将传入的关键字参数以键值对的形式存储到 config 字典中。
        if kwargs:
            for attr, val in kwargs.items():
                self.config[attr] = val
        #调用原始函数，传入相同的参数，将装饰器的功能应用到原始函数上。
        return func(self, *args, **kwargs)
    return wrapper

#简化 PyTorch 模型的加载过程，使加载过程不需要手动指定模型架构的配置信息。
#这个基类具有一些功能，包括保存模型和加载模型。
class LoadableModel(nn.Module):
    """
    Base class for easy pytorch model loading without having to manually
    specify the architecture configuration at load time.

    We can cache the arguments used to the construct the initial network, so that
    we can construct the exact same network when loading from file. The arguments
    provided to __init__ are automatically saved into the object (in self.config)
    if the __init__ method is decorated with the @store_config_args utility.
    """

    # this constructor just functions as a check to make sure that every
    # LoadableModel subclass has provided an internal config parameter
    # either manually or via store_config_args
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'config'):
            raise RuntimeError('models that inherit from LoadableModel must decorate the '
                               'constructor with @store_config_args')
        super().__init__(*args, **kwargs)

    def save(self, path):
        """
        Saves the model configuration and weights to a pytorch file.
        """
        # don't save the transformer_grid buffers - see SpatialTransformer doc for more info
        sd = self.state_dict().copy()
        grid_buffers = [key for key in sd.keys() if key.endswith('.grid')]
        for key in grid_buffers:
            sd.pop(key)
        torch.save({'config': self.config, 'model_state': sd}, path)

    @classmethod
    def load(cls, path, device):
        """
        Load a python model configuration and weights.
        """
        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        return model

import argparse
import importlib
import inspect

DATA_CLASS_MODULE = "src.data"
MODEL_CLASS_MODULE = "src.models"
LIT_MODEL_CLASS_MODULE = "src.lit_models"


def import_class(module_and_class_name: str) -> type:
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def setup_data_and_model_from_args(args: argparse.Namespace):
    data_class = import_class(f"{DATA_CLASS_MODULE}.{args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{args.model_class}")

    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)

    return data, model

def _get_init_arguments_and_types(cls):
    cls_default_params = inspect.signature(cls).parameters
    name_type_default = []
    for arg in cls_default_params:
        arg_type = cls_default_params[arg].annotation
        arg_default = cls_default_params[arg].default
        try:
            if type(arg_type).__name__ == "_LiteralGenericAlias":
                # Special case: Literal[a, b, c, ...]
                arg_types = tuple({type(a) for a in arg_type.__args__})
            elif "typing.Literal" in str(arg_type) or "typing_extensions.Literal" in str(arg_type):
                # Special case: Union[Literal, ...]
                arg_types = tuple({type(a) for union_args in arg_type.__args__ for a in union_args.__args__})
            else:
                # Special case: ComposedType[type0, type1, ...]
                arg_types = tuple(arg_type.__args__)
        except (AttributeError, TypeError):
            arg_types = (arg_type,)

        name_type_default.append((arg, arg_types, arg_default))

    return name_type_default

def _parse_argparser(cls, arg_parser: argparse.Namespace) -> argparse.Namespace:
    """Parse CLI arguments, required for custom bool types."""
    args = arg_parser.parse_args() if isinstance(arg_parser, argparse.ArgumentParser) else arg_parser

    types_default = {arg: (arg_types, arg_default) for arg, arg_types, arg_default in _get_init_arguments_and_types(cls)}

    modified_args = {}
    for k, v in vars(args).items():
        if k in types_default and v is None:
            arg_types, arg_default = types_default[k]
            if bool in arg_types and isinstance(arg_default, bool):
                v = True

        modified_args[k] = v
    return argparse.Namespace(**modified_args)

def from_argparse_args(cls, args, **kwargs):

    if isinstance(args, argparse.ArgumentParser):
        args = _parse_argparser(cls, args)

    params = vars(args)

    valid_kwargs = inspect.signature(cls.__init__).parameters
    trainer_kwargs = {name: params[name] for name in valid_kwargs if name in params}
    trainer_kwargs.update(**kwargs)

    return trainer_kwargs
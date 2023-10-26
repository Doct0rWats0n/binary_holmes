import argparse
import importlib
import lightning.pytorch as pl
from lightning.pytorch import callbacks as callbacks


from training.utils import (
    DATA_CLASS_MODULE,
    LIT_MODEL_CLASS_MODULE,
    MODEL_CLASS_MODULE,
    import_class,
    setup_data_and_model_from_args,
    from_argparse_args
)

import warnings
warnings.filterwarnings("ignore")

def _setup_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--data_class",
        type=str,
        default="MNIST",
        help=f"String identifier for the data class, relative to {DATA_CLASS_MODULE}.",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="MLP",
        help=f"String identifier for the model class, relative to {MODEL_CLASS_MODULE}.",
    )

    parser.add_argument(
        "--lit_model_class",
        type=str,
        default="BaseLitModel",
        help=f"String identifier for the lit model wrapper class, relative to {LIT_MODEL_CLASS_MODULE}.",
    )

    parser.add_argument(
        "--load_checkpoint", type=str, default=None
    )

    parser.add_argument(
        "--stop_early",
        type=int,
        default=0,
    )

    trainer_args = parser.add_argument_group("Trainer Args")
    trainer_args.add_argument(
        "--accelerator",
        type=str,
        default="gpu"
    )
    trainer_args.add_argument(
        "--devices",
        nargs="+",
        type=int
    )
    trainer_args.add_argument(
        "--max_epochs",
        type=int,
        default=5
    )
    trainer_args.add_argument(
        "--min_epochs",
        type=int,
        default=1
    )
    
    experiment_args = parser.add_argument_group("Experiment Args")
    experiment_args.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of Experiment for wandb"
    )
    experiment_args.add_argument(
        "--wandb",
        action="store_true"
    )
    experiment_args.add_argument(
        "--monitor",
        type=str,
        default="validation/loss"
    )
    experiment_args.add_argument(
        "--mode",
        type=str,
        default="min"
    )
    experiment_args.add_argument(
        "--patience",
        type=int,
        default=5
    )

    temp_args, _ = parser.parse_known_args()
    data_class = import_class(f"{DATA_CLASS_MODULE}.{temp_args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{temp_args.model_class}")
    lit_model_class = import_class(f"{LIT_MODEL_CLASS_MODULE}.{temp_args.lit_model_class}")

    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_model_class.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    data, model = setup_data_and_model_from_args(args)

    lit_model_class = import_class(f"{LIT_MODEL_CLASS_MODULE}.{args.lit_model_class}")

    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(model=model, args=args)

    checkpoint_callback = callbacks.ModelCheckpoint(
        save_top_k=3,
        filename="epoch={epoch:04d}-validation.loss={validation/loss:.3f}",
        monitor=args.monitor,
        mode=args.mode,
    )

    early_stopping_callback = callbacks.EarlyStopping(
        monitor=args.monitor,
        mode=args.mode,
        patience=args.patience
    )

    if args.wandb:
        logger = pl.loggers.WandbLogger(name=args.name, project='binary_scanner')
    else:
        logger = pl.loggers.TensorBoardLogger()
    
    trainer = pl.Trainer(**from_argparse_args(pl.Trainer, args), callbacks=[checkpoint_callback, early_stopping_callback], logger=logger)
    trainer.fit(lit_model, datamodule=data)

if __name__ == "__main__":
    main()
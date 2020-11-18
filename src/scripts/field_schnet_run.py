import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import os
import schnetpack as spk
from torch.utils.data.sampler import RandomSampler
import logging

from field_schnet.utils.script_utils import evaluate_dataset
from field_schnet.utils import compute_shielding_scaling


def get_required_fields(cfg):
    fields = []
    for p in cfg.tradeoffs:
        if p in spk.Properties.electric_properties and spk.Properties.electric_field not in fields:
            fields.append(spk.Properties.electric_field)
        if p in spk.Properties.magnetic_properties and spk.Properties.magnetic_field not in fields:
            fields.append(spk.Properties.magnetic_field)

    if cfg.mode == "solvent" or cfg.mode == "qmmm":
        if spk.Properties.electric_field not in fields:
            fields.append(spk.Properties.electric_field)

    return fields


def prepare_data(cfg):
    logging.info("Loading data...")
    data = spk.data.AtomsData(hydra.utils.to_absolute_path(cfg.data_path))

    if cfg.data.split_file is not None and os.path.exists(cfg.data.split_file):
        data_train, data_val, data_test = spk.data.partitioning.train_test_split(
            data, split_file=cfg.data.split_file
        )
    else:
        data_train, data_val, data_test = spk.data.partitioning.train_test_split(
            data, num_train=cfg.data.num_train, num_val=cfg.data.num_val, split_file=cfg.data.split_file
        )

    # Create loaders
    loaders = {
        "train": spk.data.AtomsLoader(
            data_train, batch_size=cfg.data.batch_size, num_workers=4, pin_memory=cfg.cuda,
            sampler=RandomSampler(data_train)
        ),
        "val": spk.data.AtomsLoader(
            data_val, batch_size=cfg.data.val_batch_size, num_workers=2, pin_memory=cfg.cuda
        ),
        "test": spk.data.AtomsLoader(
            data_test, batch_size=cfg.data.test_batch_size, num_workers=2, pin_memory=cfg.cuda
        ),
    }

    # Compute statistics
    if cfg.mode == "train":
        logging.info("Computing statistics...")
        mean, stddev = loaders["train"].get_statistics(spk.Properties.energy, divide_by_atoms=True)
        logging.info("{:s} mean:   {:10.5f}".format(spk.Properties.energy, mean[spk.Properties.energy].numpy()[0]))
        logging.info("{:s} stddev: {:10.5f}".format(spk.Properties.energy, stddev[spk.Properties.energy].numpy()[0]))
    else:
        mean, stddev = None, None

    return loaders, mean, stddev


def prepare_model(cfg, mean=None, stddev=None, atomref=None):
    device = torch.device("cuda" if cfg.cuda else "cpu")

    if cfg.mode == "train":
        required_fields = get_required_fields(cfg)

        representation = hydra.utils.instantiate(
            cfg.model.representation,
            required_fields=required_fields
        )

        atomwise_output = spk.atomistic.Atomwise(
            cfg.model.representation.features,
            mean=mean[spk.Properties.energy],
            stddev=stddev[spk.Properties.energy],
            atomref=atomref,
        )

        model = hydra.utils.instantiate(
            cfg.model.output,
            field_representation=representation,
            energy_model=atomwise_output,
            requested_properties=list(cfg.tradeoffs)
        )
    else:
        model = torch.load(os.path.join(os.getcwd(), "best_model"), map_location=device)

    logging.info("Model has {:d} parameters".format(spk.utils.count_params(model)))

    return model.to(device)


def train(cfg, model, loaders):
    # setup optimizer for training
    to_opt = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(to_opt, lr=cfg.training.lr)

    # Training hooks
    hooks = [
        spk.train.MaxEpochHook(cfg.training.max_epochs),
        spk.train.ReduceLROnPlateauHook(
            optimizer,
            patience=cfg.training.lr_patience,
            factor=cfg.training.lr_decay,
            min_lr=1e-6,
            window_length=1,
            stop_after_min=True
        )
    ]

    metrics = []
    for p in cfg.tradeoffs:
        metrics.append(spk.metrics.MeanAbsoluteError(p, p))
        metrics.append(spk.metrics.RootMeanSquaredError(p, p))

    if cfg.training.logger == 'csv':
        logger = spk.train.hooks.CSVHook(
            os.path.join(os.getcwd(), 'log'), metrics, every_n_epochs=cfg.training.log_every_n_epochs
        )
        hooks.append(logger)
    elif cfg.training.logger == 'tensorboard':
        logger = spk.train.hooks.TensorboardHook(
            os.path.join(os.getcwd(), 'log'), metrics, every_n_epochs=cfg.training.log_every_n_epochs
        )
        hooks.append(logger)

    device = torch.device("cuda" if cfg.cuda else "cpu")

    # Compute scaling for shielding Tensors (for each element)
    if spk.Properties.shielding in cfg.tradeoffs:
        logging.info("Computing NMR scaling for loss function...")
        shielding_weights = compute_shielding_scaling(loaders["train"], device=device)
    else:
        shielding_weights = None

    # setup loss function
    def loss(batch, result):

        err_sq = 0.0
        for p in cfg.tradeoffs:
            p_diff = batch[p] - result[p]

            if p == spk.Properties.shielding and shielding_weights is not None:
                p_diff = p_diff * shielding_weights[batch['_atomic_numbers'].long()][:, :, None, None]

            p_diff = p_diff ** 2
            p_err = torch.mean(p_diff.view(-1))
            err_sq = err_sq + cfg.tradeoffs[p] * p_err

        return err_sq

    trainer = spk.train.Trainer(
        os.getcwd(),
        model,
        loss,
        optimizer,
        loaders["train"],
        loaders["val"],
        hooks=hooks
    )

    logging.info("Training on {:s}...".format(("CPU", "GPU")[cfg.cuda]))
    trainer.train(device)


def eval(cfg, model, loaders):
    # Construct metrics and header
    header = ['Subset']
    metrics = []
    for p in cfg.tradeoffs:
        header += [f"MAE_{p}", f"RMSE_{p}"]
        metrics.append(spk.metrics.MeanAbsoluteError(p, p))
        metrics.append(spk.metrics.RootMeanSquaredError(p, p))

    split = cfg.evaluation.split
    device = torch.device("cuda" if cfg.cuda else "cpu")

    results = []

    if split == "traim" or split == "all":
        logging.info("Evaluating training error...")
        results.append(['training'] + ['%.5f' % i for i in evaluate_dataset(metrics, model, loaders["train"], device)])
    if split == "val" or split == "all":
        logging.info("Evaluating validation error...")
        results.append(['validation'] + ['%.5f' % i for i in evaluate_dataset(metrics, model, loaders["val"], device)])
    if split == "test" or split == "all":
        logging.info("Evaluating test error...")
        results.append(['test'] + ['%.5f' % i for i in evaluate_dataset(metrics, model, loaders["test"], device)])

    header = ','.join(header)
    results = np.array(results)

    np.savetxt(os.path.join(os.getcwd(), 'evaluation.csv'), results, header=header, fmt='%s', delimiter=',')


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Load custom config and use to update defaults
    if cfg.load_config is not None:
        config_path = hydra.utils.to_absolute_path(cfg.load_config)
        logging.info("Loading config from {:s}".format(config_path))
        loaded_config = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(cfg, loaded_config)

    # Load data and compute statistics
    loaders, mean, stddev = prepare_data(cfg)

    # Prepare model
    model = prepare_model(cfg, mean=mean, stddev=stddev)

    # Training mode
    if cfg.mode == "train":
        train(cfg, model, loaders)
    # Evaluation mode
    elif cfg.mode == "eval":
        eval(cfg, model, loaders)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()

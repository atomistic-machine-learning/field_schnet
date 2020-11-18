from tqdm import tqdm

__all__ = [
    "evaluate_dataset"
]


def evaluate_dataset(metrics, model, loader, device):
    """
    Perform evaluation on a single dataset. If requested, the procedure can be repeated multiple times to account
    for random sampling effects.

    Args:
        metrics:
        model (torch.model): Trained model.
        loader (schnetpack.data.AtomsLoader): Data laoder for data split.
        device (torch.device): Computation device.

    Returns:
        dict: Dictionary of evaluation results.
    """

    for metric in metrics:
        metric.reset()

    for batch in tqdm(loader):
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }
        result = model(batch)

        for metric in metrics:
            metric.add_batch(batch, result)

    results = [
        metric.aggregate() for metric in metrics
    ]
    return results

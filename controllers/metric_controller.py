def metric_fn(items, config):
    metrics = [item["metrics"] for item in items]
    if config["display_metric"] == "accuracy":
        correct_predictions = sum([metric["correct_predictions"] for metric in metrics])
        total = sum([metric["total"] for metric in metrics])
        accuracy = correct_predictions / total if total > 0 else 0
        loss = sum([metric["loss"] for metric in metrics]) / len(metrics) if len(metrics) > 0 else 0

        if "last_total" in metrics[0]:
            correct_predictions = sum([metric["last_correct_predictions"] for metric in metrics])
            total = sum([metric["last_total"] for metric in metrics])
            last_accuracy = correct_predictions / total if total > 0 else 0
            composed_metric = {"loss": loss,
                               "last_accuracy": last_accuracy * 100,
                               "accuracy": accuracy * 100}
        else:
            composed_metric = {"loss": loss,
                               "accuracy": accuracy * 100}

    return composed_metric


def compose_dev_metric(metrics, config):
    total_metric = 0
    n = len(metrics)
    for key in metrics:
        total_metric += metrics[key][config["save_by"]]
    return config["metric_direction"] * total_metric / n

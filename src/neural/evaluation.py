from collections import defaultdict


def _safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def confusion_at_threshold(rows, threshold):
    report = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    for row in rows:
        label = row["label"]
        predicted = row["probability"] >= threshold

        if label == 1 and predicted:
            report["TP"] += 1
        elif label == 1:
            report["FN"] += 1
        elif predicted:
            report["FP"] += 1
        else:
            report["TN"] += 1

    report["precision"] = _safe_divide(report["TP"], report["TP"] + report["FP"])
    report["recall"] = _safe_divide(report["TP"], report["TP"] + report["FN"])
    report["false_positive_rate"] = _safe_divide(report["FP"], report["FP"] + report["TN"])
    report["false_negative_rate"] = _safe_divide(report["FN"], report["FN"] + report["TP"])

    return report


def grouped_confusion(rows, threshold, group_key):
    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[str(row[group_key])].append(row)

    return {
        group: confusion_at_threshold(group_rows, threshold)
        for group, group_rows in grouped_rows.items()
    }

import numpy as np


#ATE and R2 metrics for benchmarking
def ate_metric(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_metric(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

plot_metrics = {
    'ATE': ate_metric,
    'R2': r2_metric
}   

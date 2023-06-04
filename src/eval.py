import numpy as np
import torch
from sksurv.metrics import integrated_brier_score


def model_predict(model, input, device="cuda:0", predict_type=None):
    model.eval()
    with torch.no_grad():
        input_ = input.to(device)
        outputs = model(input_).to(device)

    if predict_type == "cif":
        return outputs.cumsum(1)
    elif predict_type == "surv":
        return 1 - outputs.cumsum(1).sum(0)
    else:
        return outputs


# low brier score good , and less than 0.25 is "good"
def brier_score(predicted_pmf, label):
    e, t = label.squeeze().split(1, dim=-1)
    e, t = e.squeeze().numpy(), t.squeeze().numpy()

    predicted_cifs = predicted_pmf.cumsum(dim=-1)
    predicted_surv = 1 - predicted_cifs.sum(1)
    predicted_surv_combined = predicted_surv.cpu().squeeze().numpy()
    #predicted_surv_e2 = predicted_surv[:, 1, :].cpu().squeeze().numpy()

    y = np.array([(True, t) if e == 0 else (False, t) for (e, t) in zip(e, t)], dtype=[("cens", "?"), ("time", "<f8")])

    time_step = max(t) / predicted_cifs.shape[2]
    times_t = np.arange(min(t), max(t), time_step, dtype="<f8")

    print(times_t)

    score = {
        "brier_score": integrated_brier_score(y, y, predicted_surv_combined, times_t),
   #     "e2": integrated_brier_score(y, y, predicted_surv_e2, times_t),
    }

    return score


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from dataloader import CompetingRiskDataset
    from model import DeepHit
    from preprocess import preprocess_pipe

    _, _, dataset_transformed_test = preprocess_pipe(dataset_hf="Gabriel/synthetic_competing_risk")

    test_data = CompetingRiskDataset(dataset_transformed_test)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=True)

    data = iter(test_loader)

    batch, label = next(data)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DeepHit().to(device)
    model.load_state_dict(torch.load("./epoch-41.pth"))

    predicted_pmf = model_predict(model, batch)

    print(predicted_pmf)

    score = brier_score(predicted_pmf, label)

    print(score)

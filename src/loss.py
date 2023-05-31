import numpy as np
import torch


def negative_log_likelihood(pmf, label):
    """
    Compute the log likelihood loss
    This function is used to compute the survival loss
    pmf (batch, col, row)
    """
    EPSILON = 1e-8
    batch_size, nr_of_events = pmf.size(0), pmf.size(1)

    e, t = label.squeeze().split(1, dim=-1)
    e = e.squeeze()  # event
    t = t.squeeze()  # time

    is_censored = e == 0

    # Uncensored: log P(T=t, K=e, | x)
    pmf_uncensored = pmf[~is_censored]
    uncensored_times = t[~is_censored]
    nr_of_uncensored_events = len(uncensored_times)

    pmf_uncensored_at_time_t = torch.zeros(nr_of_events, nr_of_uncensored_events, dtype=torch.float32, device=pmf.device)

    for i in range(pmf_uncensored.shape[1]):
        pmf_uncensored_at_time_t[i] += pmf_uncensored[:, i, uncensored_times].diagonal()

    l1_1 = (
        torch.log((pmf_uncensored_at_time_t + EPSILON).sum(0).view(-1, 1))
        if pmf_uncensored_at_time_t.size(1)
        else torch.full((batch_size - nr_of_uncensored_events, 1), EPSILON, device=pmf.device)
    )

    # Censored: log sum P(T>t| x), eq. to log sum(1-P*(T<=t| x))
    pmf_censored = pmf[is_censored]
    censored_times = t[is_censored]

    cif = torch.cumsum(pmf_censored, dim=-1)

    cif_censored_until_time_t = torch.zeros(
        nr_of_events,
        batch_size - nr_of_uncensored_events,
        dtype=torch.float32,
        device=pmf.device,
    )

    for i in range(nr_of_events):
        cif_censored_until_time_t[i] += cif[:, i, censored_times].diagonal()

    l1_2 = (
        torch.log(1 + EPSILON - cif_censored_until_time_t.sum(0).view(-1, 1))
        if cif_censored_until_time_t.size(1)
        else torch.full((nr_of_uncensored_events, 1), EPSILON, device=pmf.device)
    )

    # Final loss
    L1 = (l1_1.sum(0) + l1_2.sum(0)) / batch_size

    return -L1


def ranking_loss(pmf_all_events, batch_label, sigma=0.1):
    device = pmf_all_events.device

    e, t = batch_label.squeeze().split(1, dim=-1)

    loss = []
    idx_durations = t.view(-1, 1)
    for i in range(pmf_all_events.shape[1]):
        pmf = pmf_all_events[:, i, :]
        y = torch.zeros_like(pmf).scatter(1, idx_durations, 1.0)  # one-hot
        batch_size = pmf.shape[0]
        ones = torch.ones((batch_size, 1), device=device)

        r = pmf.cumsum(1).matmul(y.transpose(1, 0))
        diag_r = r.diag().view(1, -1)
        r = ones.matmul(diag_r) - r

        # print(r)
        r = r.transpose(0, 1)
        mat = new_pair_rank_mat(idx_durations, e)

        R = mat * torch.exp(-r / sigma)
        l = R.mean(1, keepdim=True)
        loss.append(l)

    return sum([lo.mean() for lo in loss]).unsqueeze(0)


def new_pair_rank_mat(idx_durations, events, dtype=torch.float32):
    n = len(idx_durations)

    # Create the matrix
    mat = torch.zeros((n, n), dtype=dtype, device=idx_durations.device)

    # Create boolean masks
    dur_i = idx_durations.view(n, 1)
    dur_j = idx_durations.view(1, n)
    ev_i = events.view(n, 1)
    ev_j = events.view(1, n)

    mask_1 = (dur_i < dur_j) & (ev_j == ev_i)
    mask_2 = (dur_i <= dur_j) & (ev_j == 0)
    mask_3 = (ev_i == 0) & (ev_j == 0)

    # Apply masks to the matrix
    mat[mask_1 | mask_2] = 1
    mat[mask_3] = 0

    ind = np.diag_indices(mat.shape[0])
    mat[ind[0], ind[1]] = torch.zeros(mat.shape[0], device=idx_durations.device)

    return mat


def total_loss(outputs, labels, alpha=0.3, sigma=0.1):
    l1 = negative_log_likelihood(outputs, labels)
    l2 = ranking_loss(outputs, labels, sigma)
    total_loss = alpha * l1 + (1 - alpha) * l2
    return total_loss


if __name__ == "__main__":
    from torch.optim import Adam

    from dataloader import CompetingRiskDataset, DataLoader
    from model import DeepHit
    from preprocess import preprocess_pipe

    # dataset_transformed_train, dataset_transformed_val = preprocess_pipe(dataset_hf="Gabriel/synthetic_competing_risk")
    # training_data = CompetingRiskDataset(dataset_transformed_train)
    # training_loader = DataLoader(training_data, batch_size=256, shuffle=True)
    # model = DeepHit()
    # optimizer = Adam(model.parameters(), lr=0.001)
    # model.train(False)
    # test = iter(training_loader)
    # batch, label = next(test)
    # pmf = model(batch)
    pmf = torch.tensor(
        [
            [
                [0.0467, 0.4, 0.05, 0.0532, 0.0443, 0.08, 0.0499, 0.0509, 0.0482, 0.0482],
                [0.0449, 0.4, 0.08, 0.0580, 0.0499, 0.08, 0.0535, 0.0480, 0.0512, 0.0411],
            ],
            [
                [0.0477, 0.0477, 0.576, 0.0581, 0.0475, 0.0484, 0.0530, 0.0521, 0.05, 0.0524],
                [0.0477, 0.0560, 0.497, 0.0527, 0.0413, 0.0584, 0.0547, 0.0459, 0.05, 0.0437],
            ],
            [
                [0.0477, 0.0477, 0.0576, 0.581, 0.0475, 0.0530, 0.0484, 0.0506, 0.0506, 0.05],
                [0.0477, 0.0560, 0.0497, 0.527, 0.0413, 0.0584, 0.0547, 0.0459, 0.0437, 0.05],
            ],
            # [
            #     [0.0477, 0.0477, 0.0576, 0.0581, 0.0475, 0.0530, 0.0484, 0.0506, 0.0506, 1],
            #     [0.0477, 0.0560, 0.0497, 0.0527, 0.0413, 0.0584, 0.0547, 0.0459, 0.0437, 1],
            # ],
            # [
            #     [0.0477, 0.0477, 0.0576, 0.0581, 0.0475, 0.0530, 0.0484, 0.0506, 0.0506, 1],
            #     [0.0477, 0.0560, 0.0497, 0.0527, 0.0413, 0.0584, 0.0547, 0.0459, 0.0437, 1],
            # ],
        ]
    )

    e = torch.tensor([2, 2, 2, 2, 2])
    t = torch.tensor([3, 8, 5, 6, 4])

    label = torch.tensor([[[2, 3]], [[2, 8]], [[2, 5]]])  # [[2, 5]] [[2, 6]], [[2, 4]]
    l1 = negative_log_likelihood(pmf, label)
    l2 = ranking_loss(pmf, label)

    # total = total_loss(pmf, label)
    # print("l1: ", 0.2 * l1, l1.dtype)
    print("l2: ", 1 * l2, l2.dtype)
    # print("total: ", total)

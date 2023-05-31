import torch

# import skurv.


def model_predict(model, input, device, predict_type):
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


def eval_brier():
    pass

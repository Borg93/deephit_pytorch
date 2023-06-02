import torch
from sksurv.metrics import integrated_brier_score

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

def brier_score(predicted_pmf, label):
  e, t = label.squeeze().split(1, dim=-1)
  e,t = e.squeeze().numpy(), t.squeeze().numpy()

  predicted_cifs = predicted_pmf.cumsum(dim=-1)
  predicted_surv = 1 - predicted_cifs
  predicted_surv_e1 = predicted_surv[:,0,:].detach().squeeze().numpy()
  predicted_surv_e2 = predicted_surv[:,1,:].detach().squeeze().numpy()


  y = np.array([(True,t) if e==0 else (False,t) for (e,t) in zip(e, t)], dtype=[('cens', '?'), ('time', '<f8')])

  time_step = max(t)/len(t)
  times_t = np.arange(min(t),max(t),time_step, dtype="<f8")

  score = {"e1": integrated_brier_score(y ,y, predicted_surv_e1, times_t),
           "e2": integrated_brier_score(y ,y, predicted_surv_e2, times_t)}

  return score

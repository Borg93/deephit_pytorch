import matplotlib.pyplot as plt
import torch
from torch.optim import Adam, lr_scheduler
from tqdm.auto import tqdm


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


def train_one_epoch(epoch_number, model, training_loader, optimizer, total_fn, alpha, sigma, device):
    train_loss = []
    model.train(True)

    for batch_idx, batch in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        # Zero your gradients for every batch!
        # Make predictions for this batch
        outputs = model(inputs)

        loss = total_fn(outputs, labels, alpha, sigma)

        optimizer.zero_grad()

        loss.backward()

        # Adjust learning weights
        optimizer.step()

        batch_size = inputs.size(0)
        # Gather data and report
        train_loss.append(loss.item())

        if batch_idx % 10 == 0:
            avg_train_loss = round(sum(train_loss) / len(train_loss), 6)
            print("  Running loss: ", avg_train_loss)

    return sum(train_loss) / len(train_loss)


def val_on_epoch(model, total_fn, validation_loader, alpha, sigma, device):
    model.eval()
    val_loss = []
    for batch_idx, vdata in enumerate(validation_loader):
        with torch.no_grad():
            vinputs = vdata[0].to(device)
            vlabels = vdata[1].to(device)
            voutputs = model(vinputs)
            vloss = total_fn(voutputs, vlabels, alpha, sigma)
            val_loss.append(vloss.item())
    avg_val_loss = sum(val_loss) / len(val_loss)
    return avg_val_loss


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def train(
    model,
    epochs,
    optimizer,
    total_fn,
    training_loader,
    validation_loader,
    device,
    early_stopping_tol,
    early_stopping_min_delta,
    alpha,
    sigma,
    push_to_hub,
    scheduler
):
    epoch_number = 0

    early_stopping = EarlyStopping(tolerance=early_stopping_tol, min_delta=early_stopping_min_delta)

    first_eval_loss = val_on_epoch(model, total_fn, validation_loader, alpha, sigma, device)
    print("initial_loss: ", first_eval_loss)

    log_train = []
    log_val = []

    for epoch in tqdm(range(epochs), desc=f"{epoch_number}"):
        epoch_train_loss = train_one_epoch(
            epoch_number, model, training_loader, optimizer, total_fn, alpha, sigma, device
        )
        log_train.append((epoch_number, epoch_train_loss))

        epoch_validate_loss = val_on_epoch(model, total_fn, validation_loader, alpha, sigma, device)
        log_val.append((epoch_number, epoch_validate_loss))

        # Print epoch losses
        print(f"Epoch {epoch} | train_loss: {epoch_train_loss} | validation_loss: {epoch_validate_loss}")

        early_stopping(epoch_train_loss, epoch_validate_loss)

        if early_stopping.early_stop:
            print("We are at epoch:", epoch_number)
            checkpoint(model, f"epoch-{epoch_number}.pth")

            break

        epoch_number += 1
        scheduler.step()

    if push_to_hub:
        model.push_to_hub(repo_id="Gabriel/DeepHit", commit_message=f"Training Complete Epoch {epoch_number}")
    else:
        checkpoint(model, f"epoch-{epoch_number}.pth")

    return log_train, log_val


def plot_log(log_train, log_val):
    # Plot and label the training and validation loss values

    epoch_train, loss_train = zip(*log_train)
    epoch_val, loss_val = zip(*log_val)

    plt.plot(epoch_train, loss_train, label="Training Loss")
    plt.plot(epoch_val, loss_val, label="Validation Loss")
    plt.legend()

    # Add in a title and axes labels
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    #plt.show()
    plt.savefig(f"./output_{epoch_train[-1]}.png")


if __name__ == "__main__":
    from dataloader import CompetingRiskDataset, DataLoader
    from loss import total_loss
    from model import DeepHit
    from preprocess import preprocess_pipe

    dataset_transformed_train, dataset_transformed_val, _ = preprocess_pipe(
        dataset_hf="Gabriel/synthetic_competing_risk"
    )

    training_data = CompetingRiskDataset(dataset_transformed_train)
    validation_data = CompetingRiskDataset(dataset_transformed_val)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DeepHit().to(device)
    total_fn = total_loss

# hyperparameters
    optimizer = Adam(model.parameters(), lr=0.04)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    epochs = 100
    early_stopping_tol = 2
    early_stopping_min_delta = 0.005
    alpha = 0.2
    sigma = 0.1
    batch_train_size = 256
    batch_val_size = 256
    push_to_hub = False

    training_loader = DataLoader(training_data, batch_size=batch_train_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_val_size, shuffle=True)

    log_train, log_val = train(
        model,
        epochs,
        optimizer,
        total_fn,
        training_loader,
        validation_loader,
        device,
        early_stopping_tol,
        early_stopping_min_delta,
        alpha,
        sigma,
        push_to_hub,
        scheduler
    )

    plot_log(log_train, log_val)
    
    
# import torch
# import matplotlib.pyplot as plt

# model = torch.nn.Linear(2, 1)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer ,gamma=0.98)
# # scheduler= torch.optim.lr_scheduler.StepLR (optimizer , 30 , gamma=0.01)

# lrs = []


# for i in range(80):
#     optimizer.step()
#     lrs.append(optimizer.param_groups[0]["lr"])
# #     print("Factor = ",0.1," , Learning Rate = ",optimizer.param_groups[0]["lr"])
#     scheduler.step()

# plt.title("Decaying learning rate (LR)")
# plt.xlabel("Epochs")
# plt.ylabel("LR")
# plt.plot(lrs)

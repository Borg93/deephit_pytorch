import matplotlib.pyplot as plt
import torch
from torch.optim import Adam


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


def train_one_epoch(epoch_number, model, training_loader, optimizer, total_fn, device):
    train_loss = []
    model.train(True)

    for batch_idx, batch in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        # Zero your gradients for every batch!
        # Make predictions for this batch
        outputs = model(inputs)

        loss = total_fn(outputs, labels)

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


def test_model(model, total_fn, validation_loader, device):
    model.eval()
    val_loss = []
    for batch_idx, vdata in enumerate(validation_loader):
        with torch.no_grad():
            vinputs = vdata[0].to(device)
            vlabels = vdata[1].to(device)
            voutputs = model(vinputs)
            vloss = total_fn(voutputs, vlabels)
            val_loss.append(vloss.item())
    avg_val_loss = sum(val_loss) / len(val_loss)
    return avg_val_loss


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def train(model, epochs, optimizer, total_fn, training_loader, validation_loader, device):
    epoch_number = 0

    early_stopping = EarlyStopping(tolerance=2, min_delta=0.001)

    first_eval_loss = test_model(model, total_fn, validation_loader, device)
    print("initial_loss: ", first_eval_loss)

    log_train = []
    log_val = []

    for epoch in range(epochs):
        epoch_train_loss = train_one_epoch(epoch_number, model, training_loader, optimizer, total_fn, device)
        log_train.append((epoch_number, epoch_train_loss))

        epoch_validate_loss = test_model(model, total_fn, validation_loader, device)
        log_val.append((epoch_number, epoch_validate_loss))

        # Print epoch losses
        print(f"Epoch {epoch} | train_loss: {epoch_train_loss} | validation_loss: {epoch_validate_loss}")

        early_stopping(epoch_train_loss, epoch_validate_loss)

        if early_stopping.early_stop:
            print("We are at epoch:", epoch_number)
            checkpoint(model, f"epoch-{epoch_number}.pth")

            break

        epoch_number += 1

    checkpoint(model, f"epoch-{epoch_number}.pth")

    return log_train, log_val


def plot_log(log_train, log_val):
    # Plot and label the training and validation loss values

    epoch_train, loss_train = zip(*log_train)
    epoch_val, loss_val = zip(*log_val)

    plt.plot(epoch_train, loss_train, label="Training Loss")
    plt.plot(epoch_val, loss_val, label="Validation Loss")

    # Add in a title and axes labels
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.show()


if __name__ == "__main__":
    from dataloader import CompetingRiskDataset, DataLoader
    from loss import total_loss
    from model import DeepHit
    from preprocess import preprocess_pipe

    dataset_transformed_train, dataset_transformed_val = preprocess_pipe(dataset_hf="Gabriel/synthetic_competing_risk")

    training_data = CompetingRiskDataset(dataset_transformed_train)
    validation_data = CompetingRiskDataset(dataset_transformed_val)

    training_loader = DataLoader(training_data, batch_size=256, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=256, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 200
    model = DeepHit().to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    total_fn = total_loss

    log_train, log_val = train(model, epochs, optimizer, total_fn, training_loader, validation_loader, device)

    plot_log(log_train, log_val)

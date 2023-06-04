import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset


def equidistant_discretize_time_array(dataset, num_quantiles):
    time_array = dataset["time"]
    min_val = np.min(time_array)
    max_val = np.max(time_array)
    bin_width = (max_val - min_val) / (num_quantiles - 1)
    discretized_array = np.floor((time_array - min_val) / bin_width)
    new_value_quantiles = [min_val + i * bin_width for i in range(num_quantiles)]

    quantiles = new_value_quantiles

    new_time_array = np.copy(time_array)
    for i, category in enumerate(discretized_array):
        new_time_array[i] = category  # quantiles[int(category)]

    return {"quantile": new_time_array}


def discretize_time_array(dataset, num_quantiles):
    time_array = dataset["time"]
    sorted_array = np.sort(time_array)
    quantile_boundaries = np.linspace(0, 100, num_quantiles)
    #print(quantile_boundaries)
    new_value_quantiles = np.percentile(sorted_array, quantile_boundaries)
    new_value_quantiles = new_value_quantiles +1
    new_value_quantiles[0] = new_value_quantiles[0]-1
    new_value_quantiles[-1] = new_value_quantiles[-1] -1

    # print(new_value_quantiles)
    
    discretized_array = np.digitize(time_array, new_value_quantiles)
    
    new_time_array = np.copy(time_array)
    for i, category in enumerate(discretized_array):
        new_time_array[i] = category-1 #new_value_quantiles[int(category - 1)]
    return {"quantile": new_time_array}


def visualize_quantile(dataset, eq_quantile, quantile):
    ## Plot the histogram
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    dataset = dataset["train"]

    eq_q = np.unique(eq_quantile["quantile"])
    normal_q = np.unique(quantile["quantile"])
    quantiles = [normal_q, eq_q]
    colors = ["r", "g"]
    label = ["unidistant", "equidistant"]

    for i, ax in enumerate(axes):
        # Plot histogram
        ax.hist(dataset["time"], bins=30, density=True)

        # Add vertical lines
        for q in quantiles[i]:
            ax.axvline(q, color=colors[i], linestyle="--")

        # Set labels and title for each subplot
        ax.set_xlabel("time")
        ax.set_yticks([])
        ax.set_title(label[i])

    ## Show the plot
    plt.show()

def turn_to_single_event(row):
  if row["label"] == 2:
    row["label"] = 1
  return row

def preprocess_pipe(dataset_hf="Gabriel/synthetic_competing_risk", competing_risks=True):
    dataset_competing_risk = load_dataset(
        "parquet",
        data_files={
            "train": ".cache/train.parquet",
            "validation": ".cache/val.parquet",
            "test": ".cache/test.parquet",
        },
    )

    dataset_train = dataset_competing_risk["train"]  # .select(range(2000))
    dataset_test = dataset_competing_risk["test"]  # .select(range(2000))
    dataset_val = dataset_competing_risk["validation"]  # .select(range(2000)

    if not competing_risks:
      dataset_train = dataset_train.map(turn_to_single_event)
      dataset_test = dataset_test.map(turn_to_single_event)
      dataset_val = dataset_val.map(turn_to_single_event)

    dataset_transformed_train = dataset_train.map(
        equidistant_discretize_time_array,
        batched=True,
        fn_kwargs={"num_quantiles": 10},
        batch_size=len(dataset_train),
    )
    dataset_transformed_val = dataset_val.map(
        equidistant_discretize_time_array,
        batched=True,
        fn_kwargs={"num_quantiles": 10},
        batch_size=len(dataset_val),
    )

    dataset_transformed_test = dataset_test.map(
        equidistant_discretize_time_array,
        batched=True,
        fn_kwargs={"num_quantiles": 10},
        batch_size=len(dataset_test),
    )

    return dataset_transformed_train, dataset_transformed_val, dataset_transformed_test


if __name__ == "__main__":
    dataset_transformed_train, dataset_transformed_val, dataset_transformed_test = preprocess_pipe(
        dataset_hf="Gabriel/synthetic_competing_risk"
    )

    unique_train, counts_train = np.unique(dataset_transformed_train["quantile"], return_counts=True)
    unique_val, counts_val = np.unique(dataset_transformed_val["quantile"], return_counts=True)

    print(unique_train, counts_train)
    print(unique_val, counts_val)

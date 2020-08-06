"""
    Module contains the implementation of some
    model evaluation procedure.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def calculate_mean_and_stdd(list_of_values):
    """
    Calculates the mean and standard deviation of a set
    :param list_of_values:
    :return: mean and standard deviation
    """
    mean = 0
    stdd = 0
    set_size = len(list_of_values)
    for acc in list_of_values:
        mean += acc
    mean = mean/set_size
    for acc in list_of_values:
        stdd += np.power(acc - mean, 2)
    stdd = np.sqrt(stdd/set_size)

    return mean, stdd


def save_results_into_filesystem(experiments_folder, model_folder, super_exp_folder, sub_exp_folder,
                                 window_file, H, test_data, test_labels, batch_size, model):
    if not os.path.exists(experiments_folder+model_folder):
        os.mkdir(experiments_folder+model_folder)

    if not os.path.exists(super_exp_folder):
        os.mkdir(super_exp_folder)

    if not os.path.exists(sub_exp_folder):
        os.mkdir(sub_exp_folder)
        # if window_size != "":
        #   os.mkdir(sub_exp_folder+"window_"+window_size+"/")

    train_mean_acc, train_stdd_acc = calculate_mean_and_stdd(H.history["acc"])
    val_mean_acc, val_stdd_acc = calculate_mean_and_stdd(H.history["val_acc"])

    with open(window_file + "eval.txt", 'w') as f:
        predictions = model.predict(test_data, batch_size=batch_size)
        value = classification_report(test_labels.argmax(axis=1),
                                      predictions.argmax(axis=1))
        value += "\nTrain acc mean: "+str(train_mean_acc)+"\t ,Train acc stdd: "+str(train_stdd_acc)
        value += ("\nValidation acc mean: " + str(val_mean_acc) +
                  "\t ,Validation acc stdd: " + str(val_stdd_acc) + "\n\n")
        print(value)
        f.write(value)
        f.close()


def evaluate_model(test_data, test_labels, batch_size, model, n_epochs, H, n_classes, experiments_folder,
                   dataset_name, sub_dataset_name, model_folder, window_size="", save_results=False):
    ## Evaluating model
    print("[INFO] Evaluating Network")
    super_exp_folder = experiments_folder + model_folder + dataset_name
    sub_exp_folder = experiments_folder + model_folder + dataset_name + sub_dataset_name
    window_file = sub_exp_folder+"window_"+window_size+"_"

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, n_epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n_epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, n_epochs), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, n_epochs), H.history["val_acc"], label="val_acc")
    if window_size != "":
        plt.title("Training Loss and Accuracy Window "+window_size)
    else:
        plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()

    if save_results:
        plt.savefig(window_file + "LossAccComparison.png")
        plt.close('all')

    # plt.show()

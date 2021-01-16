import seaborn as sns
import matplotlib.pyplot as plt 
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import numpy as np


def display_sample(sample_images, sample_labels, sample_predictions=None, num_rows=5, num_cols=10,
                   plot_title=None, fig_size=None):
    """ display a random selection of images & corresponding labels, optionally with predictions
        The display is laid out in a grid of num_rows x num_col cells
        If sample_predictions are provided, then each cell's title displays the prediction 
        (if it matches actual) or actual/prediction if there is a mismatch
    """
    assert sample_images.shape[0] == num_rows * num_cols
    prediction_labels = tf.constant((sample_predictions > 0) * 1, dtype=tf.int32)
    # a dict to help encode/decode the labels
    FASHION_LABELS = {
        0: 'Closed',
        1: 'Opened'
    }

    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=1.1)
        sns.set_style({"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]})

        f, ax = plt.subplots(num_rows, num_cols, figsize=((14, 9) if fig_size is None else fig_size),
            gridspec_kw={"wspace": 0.02, "hspace": 0.30}, squeeze=True)

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")
                # show selected image
                ax[r, c].imshow(sample_images[image_index])

                if sample_predictions is None:
                    # show the actual labels in the cell title
                    title = ax[r, c].set_title("%s" % FASHION_LABELS[sample_labels[image_index]])
                else:
                    # else check if prediction matches actual value
                    true_label = sample_labels[image_index]
                    pred_value = sample_predictions[image_index]
                    pred_label = tf.constant((pred_value > 0.5) * 1, dtype=tf.int32)
                    prediction_matches_true = (true_label == pred_label)
                    if prediction_matches_true:
                        # if actual == prediction, cell title is prediction shown in green font
                        title = pred_value - 0.5
                        title_color = 'g'   
                    else:
                        # if actual != prediction, cell title is actua/prediction in red font
                        title = pred_value - 0.5
                        title_color = 'r'
                    # display cell title
                    title = ax[r, c].set_title(title)
                    plt.setp(title, color=title_color)
        # set plot title, if one specified
        if plot_title is not None:
            f.suptitle(plot_title)

        plt.show()
        plt.close()

def predict(model, x):
    unlabel_output = model(x, training=False).numpy()
    y = tf.constant((unlabel_output >= 0.5) * 1, dtype=tf.int32)

    return y


def eer(y, y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    fnr = 1 - tpr
    return fpr[np.nanargmin(np.absolute((fnr - fpr)))]




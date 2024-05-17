import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support

def plot_cm(y_true, y_pred, name):
    """ Plot the Confusion Matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm)
    for i in range(len(["False" ,"True"])):
        for j in range(len(["False" ,"True"])):
            plt.text(j, i, cm[i, j], color="white")
    plt.colorbar()
    plt.xticks([0, 1], ["False", "True"])
    plt.yticks([0, 1], ["False", "True"])
    ppv, sensitivity, f1, _ = precision_recall_fscore_support(np.array(y_true), np.array(y_pred), average='binary')
    npv, specificity, e1, _ = precision_recall_fscore_support(1-np.array(y_true), 1-np.array(y_pred), average='binary')
    plt.title(name + "\n(Confusion Matrix)\n[PPV=" + str(round(ppv*100,2)) + "%,Sensitivity=" + str(round(sensitivity*100,2)) + "%,F1=" + str(round(f1*100,2)) + "%]\n[NPV=" + str(round(npv*100,2)) + "%,Specificity=" + str(round(specificity*100,2)) + "%,E1=" + str(round(e1*100,2)) + "%]")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.savefig(name + "_Confusion_Matrix.png", bbox_inches="tight", dpi=300)
    plt.show()
    return cm

def plot_ROC_AUC(y_true, y_prob, name):
    """ Plot the Receiver Operating Characteristic (ROC) and the Area Under the Curve (AUC)."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", label="ROC curve (AUC = " + str(round(roc_auc*100,2)) + "%)")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.legend(loc="lower right")
    plt.title(name + "\n(ROC and AUC)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(name + "_ROC_AUC.png", bbox_inches="tight", dpi=300)
    plt.show()
    return roc_auc

def print_performance(y_true, y_pred, name):
    """ Print the performance measures."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    print("Precision (" + name + "): " + str(round(precision*100,2)) + "%")
    print("Recall (" + name + "): " + str(round(recall*100,2)) + "%")
    print("F1 (" + name + "): " + str(round(f1*100,2)) + "%")
    return precision, recall, f1

def print_results(y_test, y_pred, y_prob, name):
    """ Fit and predict a model."""
    # Plot the Confusion Matrix
    plot_cm(y_test, y_pred, name)
    # Plot Receiver Operating Characteristic and Area Under the Curve
    plot_ROC_AUC(y_test, y_prob, name)
    # Print the performance measures
    print_performance(y_test, y_pred, name)
    return

def plot_thresholds(precision, recall, f1, thresholds, threshold, max_criteria, name):
    """ Plot the Performance vs Threshold values."""
    plt.plot(thresholds, precision, "r", label="Precision")
    plt.plot(thresholds, recall, "g", label="Recall")
    plt.plot(thresholds, f1, "b", label="F1-measure")
    plt.plot(threshold, max_criteria, "ko", label= "Performance = " + str(round(max_criteria*100,2)) + "%\nThreshold = " + str(threshold))
    #plt.plot([threshold, threshold], [0, max_criteria], "k")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(name + "\n(Performance vs Threshold)")
    plt.xlabel("Threshold")
    plt.ylabel("Performance")
    plt.savefig(name + "_performance.png", bbox_inches="tight", dpi=300)
    plt.show()
    return

def plot_scores(scores, y_test, model_pred, threshold, name):
    """ Plot the scores using the best Threshold value."""
    idx = np.argsort(scores) # sort by score
    scores = np.array(scores)[idx]
    y_test = np.array(y_test)[idx]
    model_pred = np.array(model_pred)[idx]
    #idx = np.array(scores)!=1 # remove scores=1
    #scores = np.array(scores)[idx]
    #y_test = np.array(y_test)[idx]
    #model_pred = np.array(model_pred)[idx]
    # Scores TP, TN
    true_positives = []
    true_negatives = []
    i = 0
    for score, y, pred in zip(scores, y_test, model_pred):
        if y==True and pred==True:
            true_positives.append((i, score))
            i += 1
        if y==False and pred==False:
            true_negatives.append((i, score))
            i += 1
    if true_positives:
        plt.scatter(*zip(*true_positives), color="blue", marker="o", label="True Positives")
    if true_negatives:
        plt.scatter(*zip(*true_negatives), color="yellow", marker="x", label="True Negatives")
    plt.plot([0, len(scores)], [threshold, threshold], "black", label="Threshold = " + str(threshold))
    plt.xticks([])
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(name + "\n(Scores)")
    plt.xlabel("Data")
    plt.ylabel("Scores")
    plt.savefig(name + "_threshold_TP+TN.png", bbox_inches="tight", dpi=300)
    plt.show()
    # Scores FP, FN
    false_positives = []
    false_negatives = []
    i = 0
    for score, y, pred in zip(scores, y_test, model_pred):
        if y==False and pred==True:
            false_positives.append((i, score))
            i += 1
        if y==True and pred==False:
            false_negatives.append((i, score))
            i += 1
    if false_positives:
        plt.scatter(*zip(*false_positives), color="red", marker="o", label="False Positives")
    if false_negatives:
        plt.scatter(*zip(*false_negatives), color="green", marker="x", label="False Negatives")
    plt.plot([0, len(scores)], [threshold, threshold], "black", label="Threshold = " + str(threshold))
    plt.xticks([])
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(name + "\n(Scores)")
    plt.xlabel("Data")
    plt.ylabel("Scores")
    plt.savefig(name + "_threshold_FP+FN.png", bbox_inches="tight", dpi=300)
    plt.show()
    # Scores predicted true and false
    #scores_true = [(i, scores[i]) for i in range(len(scores)) if model_pred[i]==True]
    #if scores_true:
    #    plt.scatter(*zip(*scores_true), color="blue", marker="o", label="True labels")
    #scores_false = [(i, scores[i]) for i in range(len(scores)) if model_pred[i]==False]
    #if scores_false:
    #    plt.scatter(*zip(*scores_false), color="yellow", marker="x", label="False labels")
    #plt.plot([0, len(scores)], [threshold, threshold], "black", label="Threshold = " + str(threshold))
    #plt.xticks([])
    #plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    #plt.title(name + "\n(Scores)")
    #plt.xlabel("Data")
    #plt.ylabel("Scores")
    #plt.savefig(name + "_threshold.png", bbox_inches="tight", dpi=300)
    #plt.show()
    # Scores TP, TN, FP and FN
    #true_positives = [(i, scores[i]) for i in range(len(scores)) if y_test[i]==True and model_pred[i]==True]
    #if true_positives:
    #    plt.scatter(*zip(*true_positives), color="blue", marker="o", label="True Positives")
    #true_negatives = [(i, scores[i]) for i in range(len(scores)) if y_test[i]==False and model_pred[i]==False]
    #if true_negatives:
    #    plt.scatter(*zip(*true_negatives), color="yellow", marker="x", label="True Negatives")
    #plt.plot([0, len(scores)], [threshold, threshold], "black", label="Threshold = " + str(threshold))
    #plt.xticks([])
    #plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    #plt.title(name + "\n(Scores)")
    #plt.xlabel("Data")
    #plt.ylabel("Scores")
    #plt.savefig(name + "_threshold_TP+TN.png", bbox_inches="tight", dpi=300)
    #plt.show()
    #false_positives = [(i, scores[i]) for i in range(len(scores)) if y_test[i]==False and model_pred[i]==True]
    #if false_positives:
    #    plt.scatter(*zip(*false_positives), color="red", marker="o", label="False Positives")
    #false_negatives = [(i, scores[i]) for i in range(len(scores)) if y_test[i]==True and model_pred[i]==False]
    #if false_negatives:
    #    plt.scatter(*zip(*false_negatives), color="green", marker="x", label="False Negatives")
    #plt.plot([0, len(scores)], [threshold, threshold], "black", label="Threshold = " + str(threshold))
    #plt.xticks([])
    #plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    #plt.title(name + "\n(Scores)")
    #plt.xlabel("Data")
    #plt.ylabel("Scores")
    #plt.savefig(name + "_threshold_FP+FN.png", bbox_inches="tight", dpi=300)
    #plt.show()
    return

import os

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


def test_model(test_set, policy_model, device):
    policy_model.eval()

    test_y_list = []
    pred_y_list = []
    for index, data_item in enumerate(test_set):
        img = data_item["image"]
        is_anomaly = data_item['is_anomaly']

        img = img.unsqueeze(0)
        img = img.to(device)

        rt_tensor = policy_model(img)
        # pred_y = rt_tensor.detach().cpu().numpy()[:, 1]
        pred_y = torch.argmax(rt_tensor, dim=1).item()

        test_y_list.append(is_anomaly)
        pred_y_list.append(pred_y)

    roc = roc_auc_score(test_y_list, pred_y_list)
    pr = average_precision_score(test_y_list, pred_y_list)
    acc = accuracy_score(test_y_list, pred_y_list)
    policy_model.train()
    return roc, pr, acc


def save_model(model, model_name):
    file_path = os.path.join('./', model_name)
    torch.save(model.state_dict(), file_path)

def show_results(episodes_total_reward, pr_auc_history, roc_auc_history, acc_history):
    # plot total reward, pr auc and roc auc history in subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 15))
    axs[0].plot(episodes_total_reward)
    axs[0].set_title('Total reward per episode')
    axs[1].plot(pr_auc_history)
    axs[1].set_title('PR AUC per validation step')
    axs[2].plot(roc_auc_history)
    axs[2].set_title('ROC AUC per validation step')
    axs[3].plot(acc_history)
    axs[3].set_title('ACC per validation step')
    plt.show()

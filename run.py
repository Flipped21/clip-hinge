import torch
import random
import numpy as np
from datasets.data_manager import DataManager
from utils import factory
from utils.toolkit import count_parameters
import json
from datetime import datetime

def run(args):
    seed = args["seed"]
    _set_random(seed)
    _set_device(args)
    train_and_evaluate(args)

    
def train_and_evaluate(args):
    # 创建结果文件名（包含时间戳避免覆盖）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    e1 = args["init_epochs"]
    e2 = args["epochs"]
    hinge = "use_hinge" if args["use_image_hinge_loss"] else "no_hinge"
    # result_file = f'/mnt/data0/lzx/PGP/clip-hinge/results/{args["dataset"]}/results_{timestamp}.txt'
    result_file = f'/mnt/data0/lzx/PGP/clip-hinge/results/{args["dataset"]}/clip-hinge_{hinge}_[e1={e1}_e2={e2}]_{timestamp}.txt'

    data_manager = DataManager(args["dataset"], args["shuffle"], args["seed"], args["init_class"], args["increment"], args)
    args["class_order"] = data_manager._class_order
    model = factory.get_model(args["model_name"], args)

    cnn_curve = {"top1": []}
    grouped_accuracies = []  # 存储每次任务的grouped准确率
    task_accuracies = []  # 存储每个任务在每个模型上的准确率矩阵 [model_id][task_id]
    total_tasks = data_manager.nb_tasks

    for task in range(total_tasks):
        # print("All params: {}".format(count_parameters(model._network)))
        # print("Trainable params: {}".format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)
        cnn_acc = model.eval_task()
        model.after_task()

        # 保存当前任务的grouped准确率
        grouped_accuracies.append(cnn_acc["grouped"])

        # 从grouped准确率中提取每个历史任务在当前模型上的准确率
        # grouped准确率格式: {'00-09': acc, '10-19': acc, ...}
        task_accs = []
        class_start = 0
        for prev_task in range(task + 1):
            task_size = data_manager.get_task_size(prev_task)
            class_end = class_start + task_size - 1
            label = f'{class_start:02d}-{class_end:02d}'
            if label in cnn_acc["grouped"]:
                task_accs.append(cnn_acc["grouped"][label])
            else:
                # 如果没有找到，使用total准确率（这种情况不应该发生）
                task_accs.append(cnn_acc["top1"])
            class_start += task_size
        task_accuracies.append(task_accs)

        print("CNN: {}".format(cnn_acc["grouped"]))
        cnn_curve["top1"].append(cnn_acc["top1"])
        print("CNN top1 curve: {}".format(cnn_curve['top1']))
    

    # 实验结束后，将结果写入文件
    with open(result_file, 'w', encoding='utf-8') as f:
        # 写入参数信息
        f.write("=== 实验参数 ===\n")
        f.write(f"数据集：{args['dataset']}\n")
        f.write(f"初始 epoch 数：{args['init_epochs']}\n")
        f.write(f"后续 epoch 数：{args['epochs']}\n")
        f.write(f"bs:{args['batch_size']}\n")
        f.write(f"prompt_length:{args['prompt_length']}\n")
        # f.write(f"使用图像Adapter:是\n")
        
        # Hinge loss相关参数
        if 'text_use_hinge_loss' :
            f.write(f"是否使用Hinge Loss:{args.get('text_use_hinge_loss', False)}\n")
            f.write(f"Hinge Margin:{args.get('text_hinge_margin', 0.2)}\n")
            f.write(f"Hinge Weight:{args.get('text_hinge_weight', 1.0)}\n")
            f.write(f"Hard Pair Threshold:{args.get('text_hard_pair_threshold', 0.8)}\n")
            f.write(f"Top Hard Pairs:{args.get('text_top_hard_pairs', 20)}\n")
        
        if 'use_image_hinge_loss' :
            f.write(f"是否使用Hinge Loss:{args.get('use_image_hinge_loss', False)}\n")
            f.write(f"Hinge Margin:{args.get('image_hinge_margin', 0.2)}\n")
            f.write(f"Hinge Weight:{args.get('image_hinge_weight', 1.0)}\n")
            f.write(f"Hard Pair Threshold:{args.get('image_hard_pair_threshold', 0.8)}\n")
            f.write(f"Top Hard Pairs:{args.get('image_top_hard_pairs', 20)}\n")
        
        f.write(f"类别回放特征数目:{args.get('samples_per_old_class', 40)}\n")
        f.write("\n\n")
        
        # 写入每次任务的grouped准确率
        f.write("=== 每次任务的Grouped准确率 ===\n")
        for i, acc in enumerate(grouped_accuracies):
            f.write(f"任务 {i+1}: {acc}\n")
        f.write("\n")
        
        # 写入最终的top1曲线
        f.write("=== 最终Top1准确率曲线 ===\n")
        f.write(f"{cnn_curve['top1']}\n")
        f.write("\n")
        
        # 计算并写入模型指标
        f.write("=== 模型指标 ===\n")
        
        # 1. AIA (Average Incremental Accuracy): 对top1准确率曲线的10个指标求平均
        if len(cnn_curve['top1']) == total_tasks:
            aia = sum(cnn_curve['top1']) / len(cnn_curve['top1'])
            f.write(f"AIA (Average Incremental Accuracy): {aia:.2f}\n")
        
        # 2. AA (Average Accuracy): 取出top1准确率曲线的最后一个指标
        if len(cnn_curve['top1']) > 0:
            aa = cnn_curve['top1'][-1]
            f.write(f"AA (Average Accuracy): {aa:.2f}\n")
        
        # 3. Forgetting: 根据公式计算
        if len(task_accuracies) == total_tasks and total_tasks > 1:
            # task_accuracies[model_id][task_id] 存储第model_id+1个模型在第task_id+1个任务上的准确率
            # 公式: Forgetting = (1 / (T - 1)) * Σ_{i=1}^{T-1} [A_{T,i} - max_{j∈[i, T-1]} A_{j,i}]
            # 其中 i 从1到T-1（任务索引，从1开始）
            T = total_tasks
            forgetting_sum = 0.0
            
            for i in range(1, T):  # i 从1到T-1（公式中的任务索引）
                # A_{T,i}: 第i个任务在第T个模型上的准确率
                # task_accuracies[T-1][i-1] 对应 A_{T, i}
                A_T_i = task_accuracies[T - 1][i - 1]
                
                # max_{j∈[i, T-1]} A_{j,i}: 第i个任务在模型i到T-1上的最大准确率
                # task_accuracies[j-1][i-1] for j in range(i, T) 对应 A_{j, i} for j in [i, T-1]
                max_A_j_i = max([task_accuracies[j - 1][i - 1] for j in range(i, T)])
                
                forgetting_sum += (A_T_i - max_A_j_i)
            
            forgetting = forgetting_sum / (T - 1)
            f.write(f"Forgetting: {forgetting:.2f}\n")
        
        f.write("\n")
        
    print(f"实验结果已保存到: {result_file}")

def _set_device(args):
    device_type = args["device"]
    gpus = []
    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))
        gpus.append(device)
    args["device"] = gpus


def _set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    print("Seed Initialized!")

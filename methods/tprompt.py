import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from methods.base import BaseLearner
from models.tclip import Ticlip
from utils.toolkit import tensor2numpy
from models.clip import clip


class TPrompts(BaseLearner):
    def __init__(self, args):
        super(TPrompts, self).__init__(args)
        self._network = Ticlip(args)
        self.args = args
        self.EPSILON = args["EPSILON"]
        self.init_epochs = args["init_epochs"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lr = args["lr"]
        self.lr_decay = args["lr_decay"]
        self.weight_decay = args["weight_decay"]
        self.batch_size = args["batch_size"]
        self.num_workers = args["num_workers"]
        self.topk = 2
        self.class_num = self._network.class_num  # 每个任务的类别数  10/20
        self.all_keys = []  # 未使用，保留占位
        
        # 文本端 Hinge loss 相关参数（TEXT-Hinge）
        self.text_use_hinge_loss = args.get("text_use_hinge_loss", True)  # 兼容旧字段
        self.text_hinge_margin = args.get("text_hinge_margin", 0.1)  # hinge loss的margin
        self.text_hinge_weight = args.get("text_hinge_weight", 1.0)  # hinge loss的权重
        self.text_hard_pair_threshold = args.get("text_hard_pair_threshold", 0.865)  # text-hard-pairs 距离阈值
        self.text_top_hard_pairs = args.get("text_top_hard_pairs", 10)  # 选取的text-hard-pairs数量
        # 是否允许旧任务的 text prompt 参与更新（用于 hinge loss）
        self.update_old_text_prompts = args.get("update_old_text_prompts", False)

        # 图像端 Hinge loss 相关参数（IMAGE-Hinge，基于类别图像均值构造 image-hard-pairs）
        self.use_image_hinge_loss = args.get("use_image_hinge_loss", True)
        # 若未指定，默认与文本端的阈值/数量保持一致
        self.image_hinge_margin = args.get("image_hinge_margin", 0.1)
        self.image_hinge_weight = args.get("image_hinge_weight", 1.0)
        self.image_hard_pair_threshold = args.get("image_hard_pair_threshold", 0.865)
        self.image_top_hard_pairs = args.get("image_top_hard_pairs", 10)
        self.samples_per_old_class = args.get("samples_per_old_class", 40)  # 0表示不使用采样特征

        # 二阶段分类头训练超参
        self.second_stage_epochs = args.get("second_stage_epochs", 0)
        self.second_stage_lr = args.get("second_stage_lr", 1e-3)
        self.second_stage_weight_decay = args.get("second_stage_weight_decay", 0.0)
        self.second_stage_momentum = args.get("second_stage_momentum", 0.9)

    def after_task(self):
        self._known_classes = self._total_classes
        
        # 保存当前任务的类别统计信息（包括Task 0，因为后续任务需要用到）
        
        self._save_current_task_statistics()

    def _save_current_task_statistics(self):
        """保存当前任务的类别统计信息"""
        network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        network.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            print("正在保存当前任务的类别均值和方差信息...\n")
            for _, (_, inputs, targets) in enumerate(self.train_loader):
                
                inputs = inputs.to(self._device)
                targets = torch.tensor(targets, dtype=torch.long).to(self._device)
                
                # 获取图像特征（使用图像 encoder ，注意这里是 adapter 之前的特征）
                image_features = network.image_encoder(inputs.type(network.dtype))
                # 注意保存的特征不要进行归一化！！！
                # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                all_features.append(image_features.cpu())
                all_labels.append(targets.cpu())
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 保存统计信息
        network.save_class_statistics(all_features, all_labels)
        
        network.train()
    
    def _compute_hard_pairs(self, new_class_ids, old_class_ids):
        """
        计算相邻类别对hard_pairs
        
        Args:
            new_class_ids: 新类别ID列表（全局ID）
            old_class_ids: 旧类别ID列表（全局ID）
            
        Returns:
            tuple: (hard_pairs列表, pairs_info列表)
                - hard_pairs: [(old_class_id, new_class_id), ...]
                - pairs_info: [(old_class_id, new_class_id, similarity, old_name, new_name), ...]
        """
        if not self.text_use_hinge_loss or len(old_class_ids) == 0:
            return [], []
        
        hard_pairs = []
        pairs_info = []
        
        # 获取所有类别的原始文本特征
        all_class_ids = old_class_ids + new_class_ids
        # 这里直接拼接 old_class_ids + new_class_ids，是为了一次性获取所有"旧+新"类别的原始文本特征（按顺序），
        # 后续 text_features[:len(old_class_ids)] 和 text_features[len(old_class_ids):] 分别对应旧、新类别的特征。
        # 这样能用一次模型前向同时取得两组特征，并便于后续构建相似度矩阵。
        network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        text_features = network.get_raw_text_features(all_class_ids)
        
        # 计算新类别和旧类别之间的相似度
        old_features = text_features[:len(old_class_ids)]  # [num_old, 512]
        new_features = text_features[len(old_class_ids):]   # [num_new, 512]
        
        # 计算余弦相似度矩阵
        similarity_matrix = torch.mm(new_features, old_features.t())  # [num_new, num_old]
        
        # 找出相似度大于阈值的pair，并收集信息
        for i, new_id in enumerate(new_class_ids):
            for j, old_id in enumerate(old_class_ids):
                similarity = similarity_matrix[i, j].item()
                if similarity > self.text_hard_pair_threshold:
                    old_name = network.get_class_name(old_id)
                    new_name = network.get_class_name(new_id)
                    pairs_info.append((old_id, new_id, similarity, old_name, new_name))
        
        # 按相似度降序排序
        pairs_info.sort(key=lambda x: x[2], reverse=True)
        
        # 只取 top_k 的 hard-pairs
        top_k = self.text_top_hard_pairs
        pairs_info = pairs_info[:top_k]
        hard_pairs = [(old_id, new_id) for old_id, new_id, _, _, _ in pairs_info]
        
        return hard_pairs, pairs_info

    def _compute_image_hard_pairs(self, train_loader, new_class_ids, old_class_ids):
        """
        基于图像端类别均值计算 hard_pairs（image-hard-pairs）
        
        - 旧类别均值：使用之前任务在 after_task 中保存的 class_means
        - 新类别均值：在当前任务训练前，对 train_loader 做一次无梯度前向统计得到
        
        Args:
            train_loader: 当前任务的训练 DataLoader（只包含新类别的真实样本）
            new_class_ids: 新类别ID列表（全局ID）
            old_class_ids: 旧类别ID列表（全局ID）
        
        Returns:
            tuple: (image_hard_pairs列表, pairs_info列表)
                - image_hard_pairs: [(old_class_id, new_class_id), ...]
                - pairs_info: [(old_class_id, new_class_id, similarity), ...]
        """
        if (not self.use_image_hinge_loss) or len(old_class_ids) == 0:
            return [], []

        network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        device = network.logit_scale.device

        # --------- 1) 统计新类别的图像均值（只在当前任务开始时做一次） ----------
        new_sum = {}
        new_cnt = {}

        network.eval()
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device) if isinstance(targets, torch.Tensor) \
                    else torch.tensor(targets, dtype=torch.long, device=self._device)

                # 只保留当前任务的新类别（通常 train_loader 本身已是新类别，但这里再做一次保险过滤）
                mask = (targets >= self._known_classes) & (targets < self._total_classes)
                if mask.sum() == 0:
                    continue
                inputs = inputs[mask]
                targets = targets[mask]

                feats = network.image_encoder(inputs.type(network.dtype))
                # 当前实现中通常不使用 adapter，这里不再额外通过 adapter
                feats = feats.detach().cpu().float()

                for f, t in zip(feats, targets.cpu()):
                    cid = int(t.item())
                    if cid not in new_sum:
                        new_sum[cid] = f.clone()
                        new_cnt[cid] = 1
                    else:
                        new_sum[cid] += f
                        new_cnt[cid] += 1

        # 只为在本任务中真正出现过的新类别计算均值
        new_means = []
        new_ids_effective = []
        for cid in new_class_ids:
            if cid in new_sum and new_cnt[cid] > 0:
                mean = (new_sum[cid] / new_cnt[cid]).to(device)
                new_means.append(mean)
                new_ids_effective.append(cid)

        if len(new_means) == 0:
            return [], []

        new_means = torch.stack(new_means, dim=0)  # [num_new, D]

        # --------- 2) 取出旧类别的图像均值（来自历史统计） ----------
        old_means = []
        old_ids_effective = []
        for cid in old_class_ids:
            # class_means 在 Ticlip 中以全局类别ID为键保存
            if cid in network.class_means:
                old_means.append(network.class_means[cid].to(device))
                old_ids_effective.append(cid)

        if len(old_means) == 0:
            return [], []

        old_means = torch.stack(old_means, dim=0)  # [num_old, D]

        # --------- 3) 计算图像均值之间的余弦相似度矩阵 ----------
        # 确保 new_means 和 old_means 的 dtype 一致（CLIP 通常使用 half precision）
        target_dtype = network.dtype if hasattr(network, "dtype") else new_means.dtype
        new_means = new_means.to(device=device, dtype=target_dtype)
        old_means = old_means.to(device=device, dtype=target_dtype)

        new_feats = F.normalize(new_means, dim=-1)
        old_feats = F.normalize(old_means, dim=-1)

        # [num_new, num_old]
        similarity_matrix = torch.mm(new_feats, old_feats.t())

        # --------- 4) 筛选 image-hard-pairs ----------
        pairs_info = []
        thr = self.image_hard_pair_threshold

        for i, new_id in enumerate(new_ids_effective):
            for j, old_id in enumerate(old_ids_effective):
                sim = similarity_matrix[i, j].item()
                if sim > thr:
                    pairs_info.append((old_id, new_id, sim))

        # 按相似度降序排序
        pairs_info.sort(key=lambda x: x[2], reverse=True)

        # 只取 top_k 的 image-hard-pairs
        top_k = self.image_top_hard_pairs
        pairs_info = pairs_info[:top_k]
        image_hard_pairs = [(old_id, new_id) for old_id, new_id, _ in pairs_info]

        return image_hard_pairs, pairs_info

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)  # 旧任务类别数+新任务类别数
        self._network.update_fc()  # self.task+=1
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), mode="train")
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        val_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), mode="test")
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        # 第一阶段训练前关闭分类头融合
        net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        net.fuse_classifier_logits = False
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        # 二阶段训练：仅从第二个任务开始
        if self._cur_task > 0 and self.second_stage_epochs > 0:
            self._train_second_stage(self.train_loader)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            if "text_prompt_pool" in name:
                # 如果使用采样特征训练，需要训练所有旧任务的文本提示
                if self._cur_task > 0 and self.samples_per_old_class > 0:
                    # 训练所有已见过的任务的文本提示
                    param.requires_grad_(True)
                elif self.update_old_text_prompts:
                    # 若配置允许，则旧任务也训练
                    param.requires_grad_(True)
                elif "text_prompt_pool" + "." + str(self._network.task - 1) in name:
                    # 当前任务一定训练
                    param.requires_grad_(True)
            if "image_adapter" in name:
                param.requires_grad_(True)

        trainable = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                trainable.add(name)
        # 统计参数数量（在设置requires_grad之后）
        from utils.toolkit import count_parameters
        all_params = count_parameters(self._network)
        trainable_params = count_parameters(self._network, True)
        print(f"All params: {all_params}")
        print(f"Trainable params: {trainable_params}")
        print(f"Parameters to be updated: {trainable}")

        if self._cur_task == 0:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr, weight_decay=self.init_weight_decay)
            schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.init_epochs)
            self.run_epoch = self.init_epochs
            self.train_function(train_loader, test_loader, optimizer, schedule)
        else:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.lr, weight_decay=self.weight_decay)
            schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.epochs)
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, schedule)

    def train_function(self, train_loader, test_loader, optimizer, schedule):
        # 计算文本端 hard_pairs（只在增量任务中使用）
        hard_pairs = []
        image_hard_pairs = []
        if self._cur_task > 0:
            # 获取新类别和旧类别的全局ID
            new_class_ids = list(range(self._known_classes, self._total_classes))
            old_class_ids = list(range(0, self._known_classes))

            # ---------- 文本端 hard-pairs ----------
            if self.text_use_hinge_loss:
                hard_pairs, pairs_info = self._compute_hard_pairs(new_class_ids, old_class_ids)
                print(f"\n{'='*80}")
                print(f"Using top-{len(hard_pairs)} TEXT hard pairs for task {self._cur_task} "
                      f"(threshold={self.text_hard_pair_threshold})")
                print(f"{'='*80}")
                if len(pairs_info) > 0:
                    print(f"{'Rank':<6} {'Old Class':<30} {'New Class':<30} {'Similarity':<12}")
                    print(f"{'-'*80}")
                    for rank, (old_id, new_id, sim, old_name, new_name) in enumerate(pairs_info, 1):
                        print(f"{rank:<6} {old_name:<30} {new_name:<30} {sim:.4f}")
                print(f"{'='*80}\n")

            # ---------- 图像端 hard-pairs（基于类别图像均值） ----------
            if self.use_image_hinge_loss:
                image_hard_pairs, image_pairs_info = self._compute_image_hard_pairs(
                    train_loader, new_class_ids, old_class_ids
                )
                print(f"\n{'='*80}")
                print(f"Using top-{len(image_hard_pairs)} IMAGE hard pairs for task {self._cur_task} "
                      f"(threshold={self.image_hard_pair_threshold})")
                print(f"{'='*80}")
                if len(image_pairs_info) > 0:
                    # 为了与文本端 hard-pairs 的打印保持一致，这里显示类别名称而不是 ID
                    print(f"{'Rank':<6} {'Old Class':<30} {'New Class':<30} {'Img-Sim':<12}")
                    print(f"{'-'*80}")
                    network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
                    for rank, (old_id, new_id, sim) in enumerate(image_pairs_info, 1):
                        old_name = network.get_class_name(old_id)
                        new_name = network.get_class_name(new_id)
                        print(f"{rank:<6} {old_name:<30} {new_name:<30} {sim:.4f}")
                print(f"{'='*80}\n")
        
        # 如果使用采样特征，在每个epoch开始时采样所有旧类别特征
        use_replay = self._cur_task > 0 and self.samples_per_old_class > 0
        
        bar = tqdm(range(self.run_epoch))
        for _, epoch in enumerate(bar):
            self._network.train()
            
            # 在每个epoch开始时，为所有旧类别采样特征
            old_replay_features = None
            old_replay_targets = None
            old_replay_index = 0  # 用于跟踪已使用的采样特征索引
            total_replay_samples = 0  # 总采样特征数量
            
            if use_replay:
                network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
                old_class_ids = list(range(0, self._known_classes))
                network.eval()
                
                # 为每个旧类别采样所有特征
                old_features_list = []
                old_targets_list = []
                with torch.no_grad():
                    for old_class_id in old_class_ids:
                        # 采样旧类别的图像特征（未归一化的）
                        old_image_features = network.sample_from_gaussian(old_class_id, num_samples=self.samples_per_old_class)
                        # 确保在正确的设备上，并转换为正确的dtype（half precision）
                        old_image_features = old_image_features.to(self._device).type(network.dtype)
                        # 经过adapter（如果启用）
                        if network.use_image_adapter:
                            old_image_features = network.image_adapter(old_image_features)
                        # 归一化
                        old_image_features = old_image_features / old_image_features.norm(dim=-1, keepdim=True)
                        old_features_list.append(old_image_features)
                        old_targets_list.append(torch.full((self.samples_per_old_class,), old_class_id, dtype=torch.long, device=self._device))
                
                # 合并所有旧类别采样特征
                if len(old_features_list) > 0:
                    old_replay_features = torch.cat(old_features_list, dim=0)  # [num_old_classes * samples_per_old_class, D]
                    old_replay_targets = torch.cat(old_targets_list, dim=0)  # [num_old_classes * samples_per_old_class]
                    total_replay_samples = len(old_replay_targets)
                    print(f"Epoch {epoch + 1}: Sampled {total_replay_samples} replay features from {len(old_class_ids)} old classes ({self.samples_per_old_class} per class)")
                
                network.train()

            # 如果使用回放，先估算有效batch数量，以便均分回放特征
            # 注意：由于有些batch可能被跳过，我们使用len(train_loader)作为估算
            # 实际回放数量会在训练过程中动态调整
            if use_replay and old_replay_features is not None:
                # 估算有效batch数量（使用train_loader的长度作为上限）
                estimated_batches = len(train_loader)
                # 计算每个batch应该回放的特征数量（均分）
                replay_per_batch = total_replay_samples // estimated_batches if estimated_batches > 0 else 0
                remaining_replay_for_last_batch = total_replay_samples % estimated_batches if estimated_batches > 0 else 0
                print(f"Epoch {epoch + 1}: Will replay approximately {replay_per_batch} features per batch (estimated {estimated_batches} batches, {total_replay_samples} total replay features)")
                if remaining_replay_for_last_batch > 0:
                    print(f"  Note: Will distribute {remaining_replay_for_last_batch} extra replay features across batches")
            else:
                replay_per_batch = 0
                estimated_batches = 0
                remaining_replay_for_last_batch = 0
            
            losses = 0.
            ce_losses = 0.
            hinge_losses = 0.
            correct, total = 0, 0
            num_batches = 0
            valid_batch_count = 0  # 用于跟踪当前是第几个有效batch
            
            for i, (tasks, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device) if isinstance(targets, torch.Tensor) else torch.tensor(targets, dtype=torch.long, device=self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)  # 返回所有非0元素的索引,确保不包含旧类的训练集
                
                # 如果全部都是旧类别的样本，则跳过本轮训练
                if len(mask) == 0:
                    continue
                
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)
                valid_batch_count += 1
                
                # 如果使用采样特征，从预采样的特征中取一部分（均分）
                if use_replay and old_replay_features is not None:
                    network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
                    
                    remaining_replay = total_replay_samples - old_replay_index
                    remaining_batches = estimated_batches - valid_batch_count + 1  # 剩余的有效batch数量（包括当前batch）
                    
                    if remaining_replay > 0 and remaining_batches > 0:
                        # 计算这个batch应该回放的特征数量（均分剩余的特征）
                        # 使用向上取整，确保所有特征都能被使用
                        num_replay = (remaining_replay + remaining_batches - 1) // remaining_batches
                        
                        # 确保不超过剩余的特征数量
                        num_replay = min(num_replay, remaining_replay)
                        
                        replay_features_batch = old_replay_features[old_replay_index:old_replay_index + num_replay]
                        replay_targets_batch = old_replay_targets[old_replay_index:old_replay_index + num_replay]
                        old_replay_index += num_replay
                        
                        # 获取新类别的图像特征
                        new_image_features = network.image_encoder(inputs.type(network.dtype))
                        if network.use_image_adapter:
                            new_image_features = network.image_adapter(new_image_features)
                        new_image_features = new_image_features / new_image_features.norm(dim=-1, keepdim=True)
                        
                        # 合并新旧特征（采样特征在前，真实数据在后）
                        all_features = torch.cat([replay_features_batch, new_image_features], dim=0)
                        all_targets = torch.cat([replay_targets_batch, targets], dim=0)
                        
                        # 计算所有类别的logits
                        logits = network.forward_all_classes(all_features)  # [N, total_classes]
                    else:
                        # 如果所有采样特征都用完了，只使用真实数据
                        new_image_features = network.image_encoder(inputs.type(network.dtype))
                        if network.use_image_adapter:
                            new_image_features = network.image_adapter(new_image_features)
                        new_image_features = new_image_features / new_image_features.norm(dim=-1, keepdim=True)
                        logits = network.forward_all_classes(new_image_features)
                        all_targets = targets
                else:
                    # 不使用采样特征，使用原来的方式
                    logits = self._network(inputs)  # [32,10]/[32,20]
                    all_targets = targets
                
                # 交叉熵损失（使用全局类别ID）
                if use_replay:
                    # 使用所有类别的logits和全局标签
                    l_ce = F.cross_entropy(logits, all_targets)
                else:
                    # 使用当前任务的logits和任务内标签
                    l_ce = F.cross_entropy(logits, targets % self.class_num)
                
                # ---------------- Hinge 损失（文本端 + 图像端） ----------------
                l_text_hinge = torch.tensor(0.0, device=self._device)
                l_image_hinge = torch.tensor(0.0, device=self._device)

                # 文本端 hinge：使用 text-hard-pairs
                if self.text_use_hinge_loss and len(hard_pairs) > 0:
                    text_hinge_list = []
                    network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
                    
                    for old_class_id, new_class_id in hard_pairs:
                        # 采样旧类别的图像特征
                        old_image_features = network.sample_from_gaussian(old_class_id, num_samples=20)
                        old_image_features = old_image_features / old_image_features.norm(dim=-1, keepdim=True)
                        
                        # 获取新类别的文本特征（使用当前任务的text_prompts，可学习）
                        new_task_id = network.task - 1  # 当前任务ID
                        new_class_local_id = new_class_id - self._known_classes  # 转换为任务内ID
                        new_text_prompt = network.text_prompt_pool[new_task_id]
                        new_prompts = new_text_prompt()  # [class_num, 77, 512]
                        new_tokenized = new_text_prompt.tokenized_prompts
                        new_text_feature = network.text_encoder(
                            new_prompts[new_class_local_id:new_class_local_id+1],
                            new_tokenized[new_class_local_id:new_class_local_id+1]
                        )
                        new_text_feature = new_text_feature / new_text_feature.norm(dim=-1, keepdim=True)
                        
                        # 获取旧类别的文本特征（使用冻结的text_prompts）
                        old_task_id = old_class_id // self.class_num
                        old_class_local_id = old_class_id % self.class_num
                        old_text_prompt = network.text_prompt_pool[old_task_id]
                        # 是否允许旧任务 text prompt 参与反向传播
                        if self.update_old_text_prompts:
                            old_prompts = old_text_prompt()  # [class_num, 77, 512]
                            old_tokenized = old_text_prompt.tokenized_prompts
                            old_text_feature = network.text_encoder(
                                old_prompts[old_class_local_id:old_class_local_id+1],
                                old_tokenized[old_class_local_id:old_class_local_id+1]
                            )
                        else:
                            with torch.no_grad():
                                old_prompts = old_text_prompt()  # [class_num, 77, 512]
                                old_tokenized = old_text_prompt.tokenized_prompts
                                old_text_feature = network.text_encoder(
                                    old_prompts[old_class_local_id:old_class_local_id+1],
                                    old_tokenized[old_class_local_id:old_class_local_id+1]
                                )
                        old_text_feature = old_text_feature / old_text_feature.norm(dim=-1, keepdim=True)
                        
                        # 确保数据类型一致（CLIP使用half precision）
                        old_image_features = old_image_features.type(new_text_feature.dtype)
                        
                        # 计算相似度
                        neg_sim = torch.mm(old_image_features, new_text_feature.t())  # [20, 1]
                        pos_sim = torch.mm(old_image_features, old_text_feature.t())  # [20, 1]
                        
                        # 计算hinge损失
                        hinge_loss = F.relu(neg_sim - pos_sim + self.text_hinge_margin).mean()
                        text_hinge_list.append(hinge_loss)
                    
                    if len(text_hinge_list) > 0:
                        l_text_hinge = torch.stack(text_hinge_list).mean()

                # 图像端 hinge：使用 image-hard-pairs（逻辑与文本端相同，只是 pair 集不同）
                if self.use_image_hinge_loss and len(image_hard_pairs) > 0:
                    image_hinge_list = []
                    network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network

                    for old_class_id, new_class_id in image_hard_pairs:
                        # 采样旧类别的图像特征
                        old_image_features = network.sample_from_gaussian(old_class_id, num_samples=20)
                        old_image_features = old_image_features / old_image_features.norm(dim=-1, keepdim=True)

                        # 获取新类别的文本特征（使用当前任务的text_prompts，可学习）
                        new_task_id = network.task - 1  # 当前任务ID
                        new_class_local_id = new_class_id - self._known_classes  # 转换为任务内ID
                        new_text_prompt = network.text_prompt_pool[new_task_id]
                        new_prompts = new_text_prompt()  # [class_num, 77, 512]
                        new_tokenized = new_text_prompt.tokenized_prompts
                        new_text_feature = network.text_encoder(
                            new_prompts[new_class_local_id:new_class_local_id+1],
                            new_tokenized[new_class_local_id:new_class_local_id+1]
                        )
                        new_text_feature = new_text_feature / new_text_feature.norm(dim=-1, keepdim=True)

                        # 获取旧类别的文本特征（使用冻结的text_prompts）
                        old_task_id = old_class_id // self.class_num
                        old_class_local_id = old_class_id % self.class_num
                        old_text_prompt = network.text_prompt_pool[old_task_id]
                        if self.update_old_text_prompts:
                            old_prompts = old_text_prompt()  # [class_num, 77, 512]
                            old_tokenized = old_text_prompt.tokenized_prompts
                            old_text_feature = network.text_encoder(
                                old_prompts[old_class_local_id:old_class_local_id+1],
                                old_tokenized[old_class_local_id:old_class_local_id+1]
                            )
                        else:
                            with torch.no_grad():
                                old_prompts = old_text_prompt()  # [class_num, 77, 512]
                                old_tokenized = old_text_prompt.tokenized_prompts
                                old_text_feature = network.text_encoder(
                                    old_prompts[old_class_local_id:old_class_local_id+1],
                                    old_tokenized[old_class_local_id:old_class_local_id+1]
                                )
                        old_text_feature = old_text_feature / old_text_feature.norm(dim=-1, keepdim=True)

                        # 确保数据类型一致（CLIP使用half precision）
                        old_image_features = old_image_features.type(new_text_feature.dtype)

                        # 计算相似度
                        neg_sim = torch.mm(old_image_features, new_text_feature.t())  # [20, 1]
                        pos_sim = torch.mm(old_image_features, old_text_feature.t())  # [20, 1]

                        # 计算hinge损失
                        hinge_loss = F.relu(neg_sim - pos_sim + self.image_hinge_margin).mean()
                        image_hinge_list.append(hinge_loss)

                    if len(image_hinge_list) > 0:
                        l_image_hinge = torch.stack(image_hinge_list).mean()
                
                # 总损失：CE + 文本 Hinge + 图像 Hinge
                loss = (
                    l_ce
                    + self.text_hinge_weight * l_text_hinge
                    + self.image_hinge_weight * l_image_hinge
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                ce_losses += l_ce.item()
                # 记录总的 Hinge 损失（文本 + 图像）
                hinge_losses += (l_text_hinge.item() + l_image_hinge.item())
                num_batches += 1
                
                # 计算准确率
                _, preds = torch.max(logits, dim=1)
                if use_replay:
                    # 使用全局类别ID
                    correct += preds.eq(all_targets.expand_as(preds)).cpu().sum()
                    total += len(all_targets)
                else:
                    # 使用任务内标签
                    correct += preds.eq((targets % self.class_num).expand_as(preds)).cpu().sum()
                    total += len(targets)
            schedule.step()
            # 检查是否所有采样特征都被使用了
            if use_replay and old_replay_features is not None:
                if old_replay_index < total_replay_samples:
                    print(f"Warning: Only used {old_replay_index}/{total_replay_samples} replay features in this epoch")
                else:
                    print(f"Epoch {epoch + 1}: Used all {total_replay_samples} replay features")
            
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)  # 当前任务的训练精度
            test_acc = self._compute_accuracy(self._network, test_loader)  # 当前任务的测试精度
            avg_batches = num_batches if num_batches > 0 else len(train_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f} (CE: {:.3f}, Hinge: {:.3f}), Train Acc {:.2f}, Test Acc {:.2f}".format(
                self._cur_task, epoch + 1, self.run_epoch, losses / avg_batches, 
                ce_losses / avg_batches, hinge_losses / avg_batches, train_acc, test_acc
            )
            bar.set_description(info)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        print("Eval Task Start.")
        for _, (tasks, inputs, targets) in enumerate(loader):
            rand_val = np.random.rand()
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            tasks = tasks.to(self._device)
            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs, tasks, rand_val)
                else:
                    outputs = self._network.interface(inputs, tasks, rand_val)
            if self.args["mode"] == "TIL":
                predicts = torch.max(outputs, dim=1)[1] + tasks * self.class_num
            elif self.args["mode"] == "CIL":
                predicts = torch.max(outputs, dim=1)[1]
            else:
                raise NotImplementedError
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets % self.class_num).sum()
            total += len(inputs)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _train_second_stage(self, train_loader):
        """第二阶段：冻结文本提示，训练图像端分类头和LoRA（如果启用）。"""
        if self.second_stage_epochs <= 0:
            return

        network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        if not getattr(network, "use_second_stage", False):
            return

        network.to(self._device)
        task_id = network.task - 1  # 当前任务ID
        network._ensure_classifier(task_id)
        classifier = network.image_classifiers[task_id]

        # 如果启用LoRA且在第二个任务（第一次使用LoRA），初始化LoRA模块
        use_lora = getattr(network, "use_lora_in_second_stage", False)
        if use_lora and network.lora_modules is None:
            network._attach_lora_modules()
            print(f"Task {self._cur_task}: LoRA模块已附加到image encoder的transformer blocks {network.lora_layers}")

        # 冻结除分类头和LoRA外的所有参数
        for param in network.parameters():
            param.requires_grad_(False)
        
        # 启用分类头参数训练
        for param in classifier.parameters():
            param.requires_grad_(True)
        
        # 如果启用LoRA，启用LoRA参数训练
        params_to_optimize = list(classifier.parameters())
        if use_lora and network.lora_modules is not None:
            for lora_module in network.lora_modules.values():
                for param in lora_module.parameters():
                    param.requires_grad_(True)
                params_to_optimize.extend(lora_module.parameters())
            print(f"Task {self._cur_task}: LoRA参数已加入优化器（rank={network.lora_rank}, layers={network.lora_layers}）")

        optimizer = optim.SGD(
            params_to_optimize,
            momentum=self.second_stage_momentum,
            lr=self.second_stage_lr,
            weight_decay=self.second_stage_weight_decay,
        )
        schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.second_stage_epochs)

        bar = tqdm(range(self.second_stage_epochs))
        for _, epoch in enumerate(bar):
            classifier.train()
            # 如果使用LoRA，确保LoRA模块处于训练模式
            if use_lora and network.lora_modules is not None:
                for lora_module in network.lora_modules.values():
                    lora_module.train()
            running_loss, correct, total = 0.0, 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device) if isinstance(targets, torch.Tensor) else torch.tensor(
                    targets, dtype=torch.long, device=self._device
                )

                # 如果使用LoRA，需要计算梯度（不使用torch.no_grad）
                # LoRA已经集成在image_encoder中，前向传播会自动使用LoRA
                if use_lora and network.lora_modules is not None:
                    feats = network.image_encoder(inputs.type(network.dtype))
                else:
                    with torch.no_grad():
                        feats = network.image_encoder(inputs.type(network.dtype))
                
                if network.use_image_adapter:
                    feats = network.image_adapter(feats)
                feats = feats / feats.norm(dim=-1, keepdim=True)

                logits = classifier(feats)
                loss = F.cross_entropy(logits, targets % self.class_num)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = torch.max(logits, dim=1)[1]
                correct += preds.eq((targets % self.class_num).expand_as(preds)).cpu().sum()
                total += len(inputs)

            schedule.step()
            avg_loss = running_loss / max(len(train_loader), 1)
            train_acc = tensor2numpy(correct) * 100 / max(total, 1)
            info = "Task {} Stage2 {}/{} => Loss {:.3f}, Train Acc {:.2f}".format(
                self._cur_task, epoch + 1, self.second_stage_epochs, avg_loss, train_acc
            )
            bar.set_description(info)

        network.mark_classifier_trained(task_id)
        # 二阶段结束后开启分类头融合（用于后续推理/评估）
        network.fuse_classifier_logits = True
        network.train()

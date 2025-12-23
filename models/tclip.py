import torch
import torch.nn as nn
import copy
import math

from models.clip.prompt_learner import cfgc, load_clip_to_cpu, TextEncoder, PromptLearner
from models.clip import clip
from datasets.class_names import cifar100_classnames, imagenet_r_classnames



class Ticlip(nn.Module):
    def __init__(self, args):
        super(Ticlip, self).__init__()
        self.args = args
        self.cfg = cfgc()
        clip_model = load_clip_to_cpu(self.cfg)
        self.clip_moedl = clip_model
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.class_means = {}  # 保存每个类别的特征均值，使用字典以类别ID为键
        self.class_covs = {}   # 保存每个类别的协方差矩阵，使用字典以类别ID为键
        # feature_dim 通过 @property 定义，不需要在这里赋值


        class_names = self.generate(args) # 类别名称列表
        self.text_prompt_pool = nn.ModuleList([
            PromptLearner(self.cfg, class_names[i], self.clip_moedl)
            for i in range(args["total_tasks"])
        ])  # Text Prompt Pool
        
        self.use_image_adapter = args["use_image_adapter"]
        # 图像 Adapter：在图像 encoder 后面添加线性层，保持输出维度不变
        # CLIP 图像 encoder 输出维度通常是 512
        image_output_dim = self.image_encoder.output_dim  #  512
        if self.use_image_adapter:
            self.image_adapter = nn.Linear(image_output_dim, image_output_dim, bias=False)# TODO
        # 初始化 adapter：初始化为单位矩阵（保持输出不变），偏置为0
        # nn.init.eye_(self.image_adapter.weight)
        # nn.init.zeros_(self.image_adapter.bias)
        # 将 adapter 转换为与 CLIP 模型相同的 dtype（half precision）
        if self.use_image_adapter:
            self.image_adapter = self.image_adapter.type(self.dtype)# TODO
        # self.image_adapter = self.image_adapter.type(self.dtype)# TODO

        self.class_num = len(class_names[0])
        self.task = 0
        
        # 保存所有类别名称（用于hard_pairs计算）
        self.all_class_names = []
        for task_classes in class_names:
            self.all_class_names.extend(task_classes)

    @property
    def feature_dim(self):
        return self.image_encoder.output_dim

    def forward(self, image):
        logits = []
        # 使用图像 encoder + adapter
        image_features = self.image_encoder(image.type(self.dtype))
        if self.use_image_adapter:
            image_features = self.image_adapter(image_features)  # 通过 adapter #TODO
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_prompts = self.text_prompt_pool[self.task - 1]  # 当前任务的 prompt_learner
        tokenized_prompts = text_prompts.tokenized_prompts  # [10,77] 
        text_features = self.text_encoder(text_prompts(), tokenized_prompts)  # 文本特征 [10,512] text_prompts():[10, 77, 512]是整个文本embdeding
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits.append(logit_scale * image_features @ text_features.t())  # [32, 10]
        return torch.cat(logits, dim=1)
    
    def forward_all_classes(self, image_features):
        """
        计算所有已见过类别的logits（用于训练时包含旧类别）
        
        Args:
            image_features: Tensor [N, D] - 已经归一化的图像特征
            
        Returns:
            Tensor [N, total_classes] - 所有类别的logits
        """
        logits = []
        logit_scale = self.logit_scale.exp()
        
        # 确保image_features的dtype与模型一致（half precision）
        image_features = image_features.type(self.dtype)
        
        # 遍历所有已见过的任务
        for task_id in range(self.task):
            text_prompt = self.text_prompt_pool[task_id]
            tokenized_prompts = text_prompt.tokenized_prompts
            text_features = self.text_encoder(text_prompt(), tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits.append(logit_scale * image_features @ text_features.t())
        
        return torch.cat(logits, dim=1)  # [N, total_classes]

    def interface(self, images, tasks, rand_val):
        logits = []
        tasks = tasks.cpu().tolist()
        # 使用图像 encoder + adapter
        image_features = self.image_encoder(images.type(self.dtype))
        if self.use_image_adapter:
            image_features = self.image_adapter(image_features)  # 通过 adapter # TODO
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        for text_prompt in self.text_prompt_pool:
            tokenized_prompts = text_prompt.tokenized_prompts
            text_features = self.text_encoder(text_prompt(), tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits.append(logit_scale * image_features @ text_features.t())
        logits = torch.cat(logits,1)
        
        selectedlogit = []
        if self.args["mode"] == "CIL":
            # 修复：self.task 在训练后会递增
            # - 初始: self.task = 0
            # - Task 0 训练后: self.task = 1 (已见过1个任务)
            # - Task 1 训练后: self.task = 2 (已见过2个任务)
            # 所以 cur_id = self.task 表示已见过的任务数
            if rand_val < 0.5:
                cur_id = self.task  # self.task 表示已见过的任务数 #TODO
            else:
                cur_id = max(tasks)
            # cur_id = max(tasks)
        for idx, ii in enumerate(tasks):
            if self.args["mode"] == "TIL":
                selectedlogit.append(logits[idx][self.class_num*ii:self.class_num*ii+self.class_num])
            if self.args["mode"] == "CIL":
                # selectedlogit.append(logits[idx][:self.class_num * cur_id + self.class_num])
                # 选择所有已见过的类别的 logits
                if rand_val < 0.5:
                    selectedlogit.append(logits[idx][:self.class_num * cur_id])
                else:
                    selectedlogit.append(logits[idx][:self.class_num * cur_id + self.class_num])#TODO
                
        selectedlogit = torch.stack(selectedlogit)

        return selectedlogit

    def query(self, images, tasks):
        tasks = tasks.cpu().tolist()
        # 使用图像 encoder（query 方法用于获取中间表示，不需要 adapter）
        _ = self.image_encoder(images.type(self.dtype))
        representations = self.image_encoder.act['rep']  # [32,197,768]
        return representations

    def update_fc(self):
        self.task += 1

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self
    
    # 保存类别统计信息
    def save_class_statistics(self, features, labels):
        """
        保存当前任务中每个类别的统计信息（均值和协方差）
        
        Args:
            features: Tensor [N, D] - 当前任务图像特征
            labels: Tensor [N] - 对应的类别标签（全局类别ID）
        """
        unique_classes = torch.unique(labels)
        for cls in unique_classes:
            cls_id = cls.item() if isinstance(cls, torch.Tensor) else int(cls)
            cls_mask = (labels == cls)
            cls_features = features[cls_mask]
            
            # 计算均值
            mean = cls_features.mean(dim=0)
            
            # 计算协方差 (添加正则化项防止奇异矩阵)
            if cls_features.size(0) > 1:  # 需要至少2个样本来计算协方差
                # 使用无偏估计（除以N-1），对于小样本更稳定
                cov = torch.cov(cls_features.T) + 1e-4 * torch.eye(
                    cls_features.size(1), device=features.device
                )
            else:
                # 如果只有一个样本，使用单位矩阵
                cov = 1e-4 * torch.eye(cls_features.size(1), device=features.device)
            
            # 使用类别ID作为键保存，确保索引正确
            self.class_means[cls_id] = mean.cpu()  # 保存到CPU以节省GPU内存
            self.class_covs[cls_id] = cov.cpu()

    def generate(self, args):
        # dataset_name = args.get("dataset_name", "cifar100").lower()  # 默认值是cifar100
        
        if args["dataset"] == "cifar100":
            temp_names = list(cifar100_classnames.values())
            class_names = []
            for i in range(args["total_tasks"]):
                class_names.append(temp_names[10 * i:10 * i + 10])
        elif args["dataset"] == "imagenet-r":
            temp_names = list(imagenet_r_classnames.values())
            class_names = []
            for i in range(args["total_tasks"]):
                class_names.append(temp_names[20 * i:20 * i + 20])
            print(f"ImageNet-R: 任务数 {args['total_tasks']}")
        else:
            raise ValueError(f"不支持的数据集")
        
        return class_names
    
    # 高斯采样方法
    def sample_from_gaussian(self, class_idx, num_samples=20, shrink=False):
        """
        从高斯分布中采样特征
        
        Args:
            class_idx: 类别索引（全局类别ID）
            num_samples: 采样数量
            shrink: 是否使用协方差收缩
            
        Returns:
            Tensor [num_samples, feature_dim] - 采样得到的特征
        """
        if class_idx not in self.class_means:
            raise ValueError(f"Class {class_idx} statistics not found. Available classes: {list(self.class_means.keys())}")
        
        mean = self.class_means[class_idx].to(self.logit_scale.device)
        cov = self.class_covs[class_idx].to(self.logit_scale.device)
        
        # 生成标准正态分布随机向量
        vec = torch.randn(num_samples, mean.shape[-1], device=mean.device)
        
        # 协方差收缩 (处理小样本情况)
        if shrink:
            cov = self.shrink_cov(cov)
        
        
        # Cholesky分解 (更稳定的方式)
        try:
            sqrt_cov = torch.linalg.cholesky(cov)
        except:
            # 分解失败时的备选方案
            sqrt_cov = torch.sqrt(torch.diag(cov).clamp(min=1e-6)).diag()
        
        # 应用线性变换: x = mean + vec @ sqrt_cov^T
        vec = vec @ sqrt_cov.t()
        vec = vec + mean
        
        return vec

    # 协方差收缩函数 (可选)
    def shrink_cov(self, cov):
        """收缩协方差矩阵以提高稳定性"""
        diag_mean = torch.mean(torch.diagonal(cov))
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0.0)
        mask = off_diag != 0.0
        off_diag_mean = (off_diag * mask).sum() / mask.sum() if mask.sum() > 0 else 0.0
        
        iden = torch.eye(cov.shape[0], device=cov.device)
        alpha1 = 1.0  # 对角线收缩系数
        alpha2 = 1.0  # 非对角线收缩系数
        
        return cov + (alpha1 * diag_mean * iden) + (alpha2 * off_diag_mean * (1 - iden))
    
    def get_class_name(self, class_id):
        """
        根据类别ID获取类别名称
        
        Args:
            class_id: 类别ID（全局ID，不是任务内ID）
            
        Returns:
            str: 类别名称
        """
        if class_id < len(self.all_class_names):
            return self.all_class_names[class_id]
        else:
            raise ValueError(f"Class ID {class_id} out of range")
    
    def get_raw_text_features(self, class_ids):
        """
        使用原始CLIP文本编码器（不使用prompt）获取文本特征
        
        Args:
            class_ids: 类别ID列表或单个类别ID
            
        Returns:
            Tensor: 文本特征 [N, 512] 或 [512]
        """
        if isinstance(class_ids, int):
            class_ids = [class_ids]
        
        # 获取类别名称
        class_names = [self.get_class_name(cid) for cid in class_ids]
        # 构建文本提示：若存在 CTXINIT，则使用 "CTXINIT + 类名"；否则仅类名
        ctx_prefix = (self.cfg.CTXINIT_0 or "").strip()
        if ctx_prefix:
            texts = [f"{ctx_prefix} {name.replace('_', ' ')}." for name in class_names]
        else:
            texts = [name.replace("_", " ") + "." for name in class_names]
        
        # Tokenize
        tokenized_texts = torch.cat([clip.tokenize(text) for text in texts]).to(self.logit_scale.device)
        
        # 使用原始CLIP模型的encode_text方法获取文本特征
        with torch.no_grad():
            text_features = self.clip_moedl.encode_text(tokenized_texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        if len(class_ids) == 1:
            return text_features.squeeze(0)
        
        return text_features
    

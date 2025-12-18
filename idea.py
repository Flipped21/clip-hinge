for t in range(T):
    losses = 0.
    # 计算当前任务的相邻类别 pairs，这个f_text 不包含文本 prompt，是原始的 clip 文本编码器
    计算相邻类别对 hard_pairs={(i,j)|i,j分别是新类和旧类的id,dist(f_text(t_i),f_text(t_j))<a} 
    for i, (tasks, inputs, targets) in enumerate(train_loader):  # 每个 batch
    
         # hinge损失模块...
         hinge_list = [] # hinge损失list
         if hard_pairs is not None:
             for (old_class_id, new_class_id) in hard_pairs:
                 # 采样旧类别的视觉特征+归一化，对每个 hard-pair 中的旧类别采样 20 次 【20,512】
                 old_image_features = sample_from_gaussian(old_class_id, num_samples=20)
                 old_image_features = old_image_features / old_image_features.norm(dim=-1, keepdim=True)
                 # 获得新/旧类别文本特征+归一化 【1,512】
                 new_text_feature = text_encoder(new_class_id + text_prompts[new_class_id])
                 new_text_feature = new_text_feature / new_text_feature.norm(dim=-1, keepdim=True)
                 with torch.no_grad():
                     old_text_feature = text_encoder(old_class_id + text_prompts[old_class_id]) # 旧的text_prompt是冻结的
                     old_text_feature = old_text_feature / old_text_feature.norm(dim=-1, keepdim=True)
                 # 相邻新类和旧类的相似度【20,1】
                 neg_sim = cosine_similarity(old_image_features, new_text_feature)
                 pos_sim = cosine_similarity(old_image_features, old_text_feature)
                 # 计算 hinge 损失【1】,只有 neg_sim 存在反向传播
                 hinge_list.append(torch.relu(neg_sim - pos_sim +m).mean())
         l_ce = F.cross_entropy(logits, targets % self.class_num)
         l_hinge = torch.stack(hinge_list).mean() if hinge_list else torch.tensor(0.0, device=device)
         loss = l_ce + hinge_weight * l_hinge
         
         # 梯度投影模块...
         
         # 反向传播 - 旧文本提示不会更新
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         
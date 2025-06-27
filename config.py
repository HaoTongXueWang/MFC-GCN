import torch

class Config:
    def __init__(self):
        # 模型参数
        self.in_channels = 48  # 初始值，加载数据后会更新
        self.hidden_channels = 32
        self.out_channels = 32
        self.projection_size = 32
        self.num_classes = 1  # 二分类问题
        self.seed = 42  # 随机种子

        # 训练参数
        self.dropout_rate = 0.6
        self.projection_hidden_ratio = 2
        self.xavier_gain = 1.414

        # MoE参数
        self.num_experts = 5
        self.moe_hidden_dim = 64
        self.expert_capacity_factor = 0.9389098762878946
        self.gating_dropout = 0.30000000000000004
        self.router_z_loss_coef = 0.009252671581253003
        self.load_balancing_loss_coef = 0.05482107795396354
        self.gate_noise = 0.1
        self.use_auxiliary_loss = True

        # 注意力参数
        self.num_heads = 4
        self.attention_dropout = 0.5

        # 训练参数
        self.learning_rate = 0.008383271706935836
        self.weight_decay = 5.348451943040686e-06
        self.num_epochs = 1000
        self.lambda_contrastive = 0.3639024386065738
        self.lambda_fusion_contrast = 0.043155692892501055
        self.lambda_decay_rate = 0.0003328186456819455
        self.temperature = 0.6193859060151916
        self.patience = 10
        self.weight_update_freq = 10
        self.pos_weight = None
        self.n_folds = 5

        # 数据增强参数
        self.drop_edge_rates = [0.3, 0.2, 0.1, 0.2, 0.2]
        self.drop_feature_rates = [0.3, 0.3, 0.2, 0.3, 0.3]
        self.hard_sample_ratio = 0.3024385145043539
        self.hard_sample_boost = 1.3887308520608477
        self.max_drop_rate = 0.9

        # 网络权重参数
        self.network_weights = [1, 1, 1, 1, 1]
        self.min_network_weight = 0.1
        self.weight_smoothing = 0.7

        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 数据路径
        self.data_paths = [
            './data/preprocessed_ppi_data.pt',
            './data/preprocessed_path_data.pt',
            './data/preprocessed_go_data.pt',
            './data/preprocessed_exp_data.pt',
            './data/preprocessed_seq_data.pt'
        ]
        self.save_path = 'best_model_5net_moe_cv.pt'
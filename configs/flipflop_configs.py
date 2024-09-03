class optimizer_config:
    def __init__(self):
        # optimizer config
        self.max_grad_norm = 1
        self.batch_size = 16
        self.train_batch_size = 16
        self.dev_batch_size = 16
        self.bucket_size_factor = 1
        self.DataParallel = False
        self.num_workers = 0
        self.weight_decay = 0.1
        self.lr = 3e-4
        self.eps = 1e-8
        self.betas = (0.9, 0.999)
        self.warmup_steps = 50
        self.training_steps = (160000//self.batch_size)
        self.epochs = 1
        self.early_stop_patience = 1000
        self.scheduler = "linearwarmup"
        self.scheduler_patience = 2
        self.scheduler_reduce_factor = 0.5
        self.optimizer = "AdamW"
        self.save_by = "accuracy"
        self.metric_direction = 1
        self.different_betas = False
        self.chunk_size = 16000
        self.display_metric = "accuracy"


class base_config(optimizer_config):
    def __init__(self):
        super().__init__()
        self.word_embd_freeze = False
        self.embedding_type = "sparse"
        self.attn_fn = "FlashMHA"
        self.batch_pair = False
        self.embd_dim = 512
        self.hidden_size = 512
        self.ffn_dim = 2048
        self.heads = 8
        self.head_dim = 64
        self.layers = 6
        self.dropout = 0.1
        self.attn_dropout = 0.1
        self.ffn_gating = False
        self.causal = True
        self.pooling = "mean"
        self.leaf_transform = False
        self.ex_halt = False

class Transformer_config(base_config):
    def __init__(self):
        super().__init__()
        self.block = "TransformerBlock"
        self.encoder_type = "TransformerEncoder"
        self.model_name = "(Transformer)"

class UT_config(Transformer_config):
    def __init__(self):
        super().__init__()
        self.layers = 10
        self.penalty_gamma = 0.1
        self.transition_based_halt = False
        self.ex_halt = False
        self.global_halt = False
        self.attn_based_halt = False
        self.mean_based_halt = False
        self.pooling = "mean"
        self.encoder_type = "UniversalTransformerEncoder"
        self.model_name = "(Universal Transformer)"

class GUT_end_config(Transformer_config):
    def __init__(self):
        super().__init__()
        self.layers = 20
        self.penalty_gamma = 0.1
        self.transition_based_halt = True
        self.ex_halt = False
        self.global_halt = True
        self.attn_based_halt = False
        self.mean_based_halt = True
        self.pooling = "end"
        self.encoder_type = "UniversalTransformerEncoder"
        self.block = "GatedTransformerBlock"
        self.ffn_gate_dim = self.ffn_dim
        self.model_name = "(Gated Universal Transformer End)"



class TLB_config(Transformer_config):
    def __init__(self):
        super().__init__()
        self.model_chunk_size = 100
        self.update_layers = 2
        self.layers = 3
        self.memory_size = 10
        self.block = "TLBBlock"
        self.encoder_type = "TLBEncoder"
        self.model_name = "(TLB)"

class GUTLB_config(TLB_config):
    def __init__(self):
        super().__init__()
        self.layers = 10
        self.penalty_gamma = 0.1
        self.transition_based_halt = True
        self.ex_halt = False
        self.global_halt = True
        self.attn_based_halt = False
        self.mean_based_halt = True
        self.ffn_gate_dim = self.ffn_dim
        self.block = "GatedTLBBlock"
        self.encoder_type = "UTLBEncoder"
        self.model_name = "(GUTLB)"
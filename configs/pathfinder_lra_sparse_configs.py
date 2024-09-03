class optimizer_config:
    def __init__(self):
        # optimizer config
        self.max_grad_norm = 1
        self.batch_size = 128
        self.train_batch_size = 128
        self.dev_batch_size = 128
        self.bucket_size_factor = 1
        self.DataParallel = False
        self.num_workers = 0
        self.weight_decay = 0
        self.lr = 2e-4
        self.eps = 1e-6
        self.betas = (0.9, 0.999)
        self.warmup_steps = 312
        self.training_steps = 125000
        self.epochs = 100
        self.early_stop_patience = 20
        self.scheduler = "linearwarmup"
        self.optimizer = "AdamW"
        self.save_by = "accuracy"
        self.metric_direction = 1
        self.different_betas = False
        self.chunk_size = -1
        self.ex_halt = False
        self.display_metric = "accuracy"


class base_config(optimizer_config):
    def __init__(self):
        super().__init__()
        self.word_embd_freeze = False
        self.embedding_type = "sparse"
        self.attn_fn = "FlashMHA"
        self.batch_pair = False
        self.embd_dim = 64
        self.hidden_size = 64
        self.ffn_dim = 128
        self.heads = 2
        self.head_dim = 32
        self.layers = 2
        self.dropout = 0.1
        self.attn_dropout = 0.1
        self.ffn_gating = False
        self.causal = False
        self.pooling = "mean"
        self.leaf_transform = False


class Transformer_config(base_config):
    def __init__(self):
        super().__init__()
        self.block = "TransformerBlock"
        self.encoder_type = "TransformerEncoder"
        self.model_name = "(Transformer)"


class SharedTransformer_config(Transformer_config):
    def __init__(self):
        super().__init__()
        self.layers = 4
        self.encoder_type = "SharedTransformerEncoder"
        self.model_name = "(Shared Transformer)"


class UT_config(Transformer_config):
    def __init__(self):
        super().__init__()
        self.layers = 4
        self.penalty_gamma = 0.1
        self.transition_based_halt = False
        self.ex_halt = False
        self.global_halt = False
        self.attn_based_halt = False
        self.mean_based_halt = False
        self.pooling = "mean"
        self.encoder_type = "UniversalTransformerEncoder"
        self.model_name = "(Universal Transformer)"

class GUT_config(Transformer_config):
    def __init__(self):
        super().__init__()
        self.layers = 4
        self.penalty_gamma = 0.1
        self.transition_based_halt = True
        self.ex_halt = False
        self.global_halt = True
        self.attn_based_halt = False
        self.mean_based_halt = True
        self.pooling = "mean"
        self.encoder_type = "UniversalTransformerEncoder"
        self.block = "GatedTransformerBlock"
        self.ffn_gate_dim = self.ffn_dim
        self.model_name = "(Gated Universal Transformer)"


class GUT_end_config(Transformer_config):
    def __init__(self):
        super().__init__()
        self.layers = 4
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


class TLB_config(base_config):
    def __init__(self):
        super().__init__()
        self.max_grad_norm = None
        self.batch_size = 512
        self.train_batch_size = 512
        self.dev_batch_size = 512
        self.bucket_size_factor = 1
        self.DataParallel = False
        self.num_workers = 0
        self.weight_decay = 0
        self.eps = 1e-9
        self.betas = (0.9, 0.98)
        self.warmup_steps = (160000 // self.batch_size)
        self.training_steps = (160000 // self.batch_size) * 200
        self.epochs = 200
        self.early_stop_patience = 200
        self.scheduler = "cosine2"
        self.attn_dropout = 0.1
        self.embd_dim = 128
        self.hidden_size = 128
        self.heads = 8
        self.dropout = 0.1
        self.head_dim = 8
        self.ffn_dim = 128
        self.layers = 2
        self.model_chunk_size = 64
        self.update_layers = 1
        self.memory_size = 5
        self.lr = 5e-4
        self.global_position_encoding = True
        self.block = "TLBBlock"
        self.encoder_type = "TLBEncoder"
        self.model_name = "(TLB)"


class GUTLB_config(TLB_config):
    def __init__(self):
        super().__init__()
        self.layers = 3
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

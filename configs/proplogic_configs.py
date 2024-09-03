class optimizer_config:
    def __init__(self):
        # optimizer config
        self.max_grad_norm = 1
        self.batch_size = 256
        self.train_batch_size = 256
        self.dev_batch_size = 256
        self.bucket_size_factor = 1
        self.DataParallel = False
        self.num_workers = 0
        self.weight_decay = 1e-1
        self.lr = 7e-4
        self.eps = 1e-8
        self.betas = (0.9, 0.98)
        self.warmup_steps = 4000
        self.training_steps = 477 * 150
        self.epochs = 150
        self.early_stop_patience = 20
        self.scheduler = "JustWarmup"
        #self.schedule_start = 10
        #self.scheduler_patience = 2
        #self.scheduler_reduce_factor = 0.5
        self.optimizer = "AdamW"
        self.save_by = "accuracy"
        self.metric_direction = 1
        self.different_betas = False
        self.chunk_size = -1
        self.display_metric = "accuracy"


class base_config(optimizer_config):
    def __init__(self):
        super().__init__()
        self.word_embd_freeze = False
        self.embedding_type = "sparse"
        self.attn_fn = "FlashMHA"
        self.batch_pair = True
        self.embd_dim = 512
        self.hidden_size = 512
        self.ffn_dim = 1024
        self.heads = 8
        self.head_dim = 64
        self.layers = 4
        self.dropout = 0.2
        self.attn_dropout = 0.5
        self.ffn_gating = False
        self.ex_halt = False
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
        self.layers = 15
        self.encoder_type = "SharedTransformerEncoder"
        self.model_name = "(Shared Transformer)"

class SharedGatedTransformer_config(Transformer_config):
    def __init__(self):
        super().__init__()
        self.layers = 15
        self.ffn_gate_dim = self.ffn_dim
        self.pooling = "end"
        self.encoder_type = "SharedTransformerEncoder"
        self.block = "GatedTransformerBlock"
        self.model_name = "(Shared  GatedTransformer)"

class UT_config(Transformer_config):
    def __init__(self):
        super().__init__()
        self.layers = 15
        self.penalty_gamma = 0
        self.transition_based_halt = False
        self.ex_halt = False
        self.global_halt = False
        self.attn_based_halt = False
        self.mean_based_halt = False
        self.pooling = "mean"
        self.encoder_type = "UniversalTransformerEncoder"
        self.model_name = "(Universal Transformer)"

class GUTX_config(Transformer_config):
    def __init__(self):
        super().__init__()
        self.layers = 15
        self.penalty_gamma = 0
        self.transition_based_halt = True
        self.ex_halt = True
        self.global_halt = False
        self.attn_based_halt = False
        self.mean_based_halt = False
        self.pooling = "mean"
        self.encoder_type = "UniversalTransformerEncoder"
        self.block = "GatedTransformerBlock"
        self.ffn_gate_dim = self.ffn_dim
        self.model_name = "(Gated Universal Transformer X)"

class GUT_end_config(Transformer_config):
    def __init__(self):
        super().__init__()
        self.layers = 15
        self.penalty_gamma = 0.0
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

class GUT_token_end_config(GUT_end_config):
    def __init__(self):
        super().__init__()
        self.global_halt = False
        self.mean_based_halt = False
        self.pooling = "end"
        self.model_name = "(Gated Universal Transformer Token End)"

class GUT_nogate_end_config(GUT_end_config):
    def __init__(self):
        super().__init__()
        self.block = "TransformerBlock"
        self.model_name = "(Gated Universal Transformer NoGate)"

class GUT_notrans_end_config(GUT_end_config):
    def __init__(self):
        super().__init__()
        self.transition_based_halt = False
        self.model_name = "(Gated Universal Transformer No Trans)"

class GUT_config(GUT_end_config):
    def __init__(self):
        super().__init__()
        self.pooling = "mean"
        self.model_name = "(Gated Universal Transformer)"

class UT_end_config(UT_config):
    def __init__(self):
        super().__init__()
        self.pooling = "end"
        self.model_name = "(Universal Transformer End)"

class GUTX_end_config(GUTX_config):
    def __init__(self):
        super().__init__()
        self.pooling = "end"
        self.model_name = "(Gated Universal Transformer X end)"

class TLB_config(Transformer_config):
    def __init__(self):
        super().__init__()
        self.model_chunk_size = 10
        self.update_layers = 1
        self.layers = 2
        self.memory_size = 10
        self.block = "TLBBlock"
        self.encoder_type = "TLBEncoder"
        self.model_name = "(TLB)"

class GUTLB_config(TLB_config):
    def __init__(self):
        super().__init__()
        self.layers = 5
        self.penalty_gamma = 0
        self.transition_based_halt = True
        self.ex_halt = False
        self.global_halt = True
        self.attn_based_halt = False
        self.mean_based_halt = True
        self.ffn_gate_dim = self.ffn_dim
        self.block = "GatedTLBBlock"
        self.encoder_type = "UTLBEncoder"
        self.model_name = "(GUTLB)"
########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import gc, math
from random import randint
from typing import List, Optional

import numpy as np
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F

import lightning as L
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only
from lightning.pytorch.strategies import DeepSpeedStrategy

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import deepspeed.runtime.lr_schedules
import wandb

RWKV_JIT_ON = True

if RWKV_JIT_ON:
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
else:
    MyModule = nn.Module
    MyFunction = lambda x: x


class TimeMixState:

    def __init__(self, shift_state: torch.Tensor, wkv_state: torch.Tensor):
        self.shift_state = shift_state
        self.wkv_state = wkv_state


class ChannelMixState:

    def __init__(self, shift_state: torch.Tensor):
        self.shift_state = shift_state


class BlockState:

    def __init__(self, time_mix_state: TimeMixState,
                 channel_mix_state: ChannelMixState):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state


class BlockStateList:

    def __init__(self, shift_states, wkv_states):
        self.wkv_states = wkv_states
        self.shift_states = shift_states

    @staticmethod
    def create(N, B, C, device, dtype):
        result = BlockStateList.empty(N, B, C, device, dtype)
        result.wkv_states[:] = 0
        result.wkv_states[:, :, :, -1] = -1e38
        result.shift_states[:] = 0
        return result

    @staticmethod
    def empty(N, B, C, device, dtype):
        wkv_states = torch.empty((N, B, C, 3),
                                 device=device,
                                 dtype=torch.float)
        shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    def __getitem__(self, layer: int):
        return BlockState(
            TimeMixState(self.shift_states[layer, 0], self.wkv_states[layer]),
            ChannelMixState(self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state.shift_state
        self.wkv_states[layer] = state.time_mix_state.wkv_state
        self.shift_states[layer, 1] = state.channel_mix_state.shift_state


from torch.utils.cpp_extension import load

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################


class RWKV_TimeMix(MyModule):

    def __init__(self, layer_id, n_layer, n_embd, dim_att):
        super().__init__()

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # fancy time_decay
            decay_speed = torch.ones(dim_att)
            for h in range(dim_att):
                decay_speed[h] = -5 + 8 * (h /
                                           (dim_att - 1))**(0.7 +
                                                            1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1
                                   for i in range(dim_att)]) * 0.5
            self.time_first = nn.Parameter(
                torch.ones(dim_att) * math.log(0.3) + zigzag)

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(
                torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(
                torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.key = nn.Linear(n_embd, dim_att, bias=False)
        self.value = nn.Linear(n_embd, dim_att, bias=False)
        self.receptance = nn.Linear(n_embd, dim_att, bias=False)
        self.output = nn.Linear(dim_att, n_embd, bias=False)

    @MyFunction
    def forward(self, x, last_state: TimeMixState):
        # Mix x with the previous timestep to produce xk, xv, xr
        xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]),
                          dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)

        sr = torch.sigmoid(r)

        y, new_wkv_state = torch.ops.rwkv.wkv(self.time_decay, self.time_first,
                                              k, v, last_state.wkv_state)
        return self.output(sr * y), TimeMixState(x[:, -1], new_wkv_state)


########################################################################################################


class RWKV_ChannelMix(MyModule):

    def __init__(self, layer_id, n_layer, n_embd, dim_ffn):
        super().__init__()

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(n_embd, dim_ffn, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(dim_ffn, n_embd, bias=False)

    @MyFunction
    def forward(self, x, last_state: ChannelMixState):
        xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]),
                          dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        return (torch.sigmoid(self.receptance(xr)) * kv,
                ChannelMixState(x[:, -1]))


########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):

    def __init__(self, layer_id, n_layer, n_embd, dim_att, dim_ffn):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = RWKV_TimeMix(layer_id, n_layer, n_embd, dim_att)
        self.ffn = RWKV_ChannelMix(layer_id, n_layer, n_embd, dim_ffn)

    def forward(self, x, last_state: BlockState):
        if self.layer_id == 0:
            x = self.ln0(x)
        att_out, att_state = self.att(
            self.ln1(x),
            last_state.time_mix_state,
        )
        x = x + att_out
        ffn_out, ffn_state = self.ffn(
            self.ln2(x),
            last_state.channel_mix_state,
        )
        x = x + ffn_out
        return x, BlockState(att_state, ffn_state)


class L2Wrap(torch.autograd.Function):

    @staticmethod
    def forward(ctx, loss, y, token_amount, currentMask):
        ctx.save_for_backward(y)
        ctx.token_amount = token_amount
        ctx.currentMask = currentMask
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        token_amount = ctx.token_amount
        # to encourage the logits to be close to 0
        factor = 1e-4 / token_amount
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        gy = gy * ctx.currentMask[:, None][None, :]
        return (grad_output, gy, None, None)


class RWKV(L.LightningModule):

    def __init__(self,
                 ctx_len: int,
                 ctx_len_cutoffs: List[int],
                 ctx_len_warmup_steps: List[int],
                 n_embd: int,
                 n_layer: int,
                 vocab_size: int,
                 grad_cp: bool,
                 lr_init: float,
                 lr_final: float = -1.0,
                 lr_period: int = -1,
                 lr_period_type: str = 'epoch',
                 warmup_steps: int = -1,
                 beta1: float = 0.9,
                 beta2: float = 0.99,
                 adam_eps: float = 1.0e-08,
                 weight_decay: float = 0.01,
                 bptt_learning: bool = True,
                 bptt_learning_range: int = -1,
                 bptt_truncated_learning: bool = False,
                 layerwise_lr: bool = True,
                 dim_att: Optional[int] = None,
                 dim_ffn: Optional[int] = None,
                 load_model: Optional[str] = None,
                 torch_set_float32_matmul_precision:str = 'high'
                 ):
        super().__init__()
        self.ctx_len = ctx_len
        self.ctx_len_cutoffs = ctx_len_cutoffs
        self.ctx_len_warmup_steps = ctx_len_warmup_steps
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.layerwise_lr = layerwise_lr
        self.grad_cp = grad_cp
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_period = lr_period
        self.lr_period_type = lr_period_type
        self.warmup_steps = warmup_steps
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        self.bptt_learning = bptt_learning
        self.bptt_learning_range = bptt_learning_range
        self.bptt_truncated_learning = bptt_truncated_learning

        dim_att = dim_att or n_embd
        dim_ffn = dim_ffn or n_embd * 4

        if torch_set_float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(torch_set_float32_matmul_precision)

        self.emb = nn.Embedding(vocab_size, n_embd)

        load(name=f"wkv_{self.ctx_len}_bf16",
             sources=["cuda/wkv_op_bf16.cpp", "cuda/wkv_cuda_bf16.cu"],
             verbose=True,
             extra_cflags=["-std=c++17", "-O3", f"-DTmax={self.ctx_len}"],
             extra_cuda_cflags=[
                 "-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60",
                 "--use_fast_math", "-O3", "-Xptxas -O3",
                 "--extra-device-vectorization", f"-DTmax={self.ctx_len}"
             ],
             is_python_module=False)

        self.blocks = nn.ModuleList([
            Block(i, n_layer, n_embd, dim_att, dim_ffn) for i in range(n_layer)
        ])

        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.load_state_dict(torch.load(load_model, map_location='cpu'))

    def configure_optimizers(self):
        if self.bptt_learning == False:
            if self.deepspeed_stage >= 2 or self.deepspeed_offload:
                print("[WARNING]: it is highly recommended to enable bptt_learning when used to deepspeed 2/3/offloading, otherwise an exception will occur when training with dataset records, larger then the configured context length ({self.ctx_len})")
        else:
            if self.trainer.num_devices > 1:
                if self.bptt_learning_range <= 0:
                    print("[WARNING]: unlimited bptt_learning_range across multiple GPU's has a performance penalty with datasets of mixed sizes due to its constant need to keep all GPU's in sync (consider using bptt_learning_range=1 instead)")
                if self.bptt_learning_range > 1:
                    # Temporary error, till better sync logic is done for mixed document sizes
                    # (lazy to support this right now, since i have no idea if anyone has a use for it)
                    raise NotImplementedError("bptt_learning_range > 1 is not supported yet")
        if self.layerwise_lr:
            lr_1x = set()
            lr_2x = set()
            lr_3x = set()
            for n, p in self.named_parameters():
                if "time_mix" in n:
                    lr_1x.add(n)
                elif "time_decay" in n:
                    lr_2x.add(n)
                elif "time_first" in n:
                    lr_3x.add(n)
                else:
                    lr_1x.add(n)
            lr_1x = sorted(list(lr_1x))
            lr_2x = sorted(list(lr_2x))
            lr_3x = sorted(list(lr_3x))
            # print('1x', lr_1x)
            # print('2x', lr_2x)
            # print('3x', lr_3x)
            param_dict = {n: p for n, p in self.named_parameters()}
            optim_groups = [
                {
                    "params": [param_dict[n] for n in lr_1x],
                    "weight_decay": 0.0,
                    "lr": 1.0 * self.lr_init
                },
                {
                    "params": [param_dict[n] for n in lr_2x],
                    "weight_decay": 0.0,
                    "lr": 2.0 * self.lr_init
                },
                {
                    "params": [param_dict[n] for n in lr_3x],
                    "weight_decay": 0.0,
                    "lr": 3.0 * self.lr_init
                },
            ]
        else:
            optim_groups = [
                {
                    "params": [p for n, p in self.named_parameters()],
                    "weight_decay": 0.0
                },
            ]

        # Set ending_lr to starting_lr, as default behavior
        starting_lr = self.lr_init
        ending_lr = self.lr_final
        if ending_lr < 0:
            ending_lr = self.lr_init

        if self.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(optim_groups,
                                         lr=starting_lr,
                                         betas=(self.beta1, self.beta2),
                                         eps=self.adam_eps,
                                         bias_correction=True,
                                         adamw_mode=False,
                                         weight_decay=self.weight_decay,
                                         amsgrad=False)
        else:
            optimizer = FusedAdam(optim_groups,
                                  lr=starting_lr,
                                  betas=(self.beta1, self.beta2),
                                  eps=self.adam_eps,
                                  bias_correction=True,
                                  adam_w_mode=False,
                                  weight_decay=self.weight_decay,
                                  amsgrad=False)
            
        # Throw if wramup_steps and lr_period are both set (not supported)
        if self.warmup_steps > 0 and self.lr_period > 0:
            raise ValueError(
                "Use either warmup_steps or lr_period, not both.")

        if self.warmup_steps > 0:
            lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
                optimizer,
                warmup_min_lr=0.2 * self.lr_init,
                warmup_max_lr=self.lr_init,
                warmup_num_steps=self.warmup_steps,
                warmup_type='linear')

            return {
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
            }

        else:
            # Skip the lr_scheduler process if lr_init and lr_final are the same
            if starting_lr == ending_lr:
                return optimizer

            # The total number of steps to perform training rate decay with
            lr_total_step = 0

            # Handle lr_period -1 default behaviour of using the max_step / max_epoch
            if self.lr_period == -1:
                # Get trainer max_step / max_epoch
                trainer_max_step = self.trainer.max_steps
                trainer_max_epoch = self.trainer.max_epochs
                if trainer_max_step > 0:
                    lr_total_step = trainer_max_step
                elif trainer_max_epoch > 0:
                    lr_total_step = trainer_max_epoch * self.num_step_per_epoch()
                else :
                    print("Warning: max_step/max_epoch not set, we would be performing lr_init to lr_final shift assuming 10 epoch")
                    lr_total_step = 10 * self.num_step_per_epoch()
            else:
                # Calculate lr_total_step based on lr_period
                if self.lr_period_type == "step":
                    lr_total_step = self.lr_period
                elif self.lr_period_type == "epoch":
                    lr_total_step = self.lr_period * self.num_step_per_epoch()
                else:
                    raise ValueError(f"lr_period_type {self.lr_period_type} not supported.")

            # Lets initialize the lr_scheduler
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor= ending_lr / starting_lr,
                total_iters=lr_total_step
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
                
    
    # We have to compute the number of steps per epoch ourselves
    # as this value is not provided directly by pytorch lightning
    # https://github.com/Lightning-AI/lightning/issues/5449#issuecomment-1501597319
    def num_step_per_epoch(self) -> int:
        # Estimated number of steps in total, added as the following
        # https://github.com/Lightning-AI/lightning/pull/11599
        #
        # This MUST be called before len(self.trainer.train_loader)
        # otherwise there is a bug in which the train_dataloader is not
        # fully initialized, which seems to be resolved by computing the 
        # self.trainer.estimated_stepping_batches
        estimated_stepping_batches = self.trainer.estimated_stepping_batches

        # Get the number of epochs, 
        # use estimated_stepping_batches if max_epochs is set
        max_epochs = self.trainer.max_epochs
        if max_epochs > 0:
            return estimated_stepping_batches // max_epochs

        # Get the train_dataloader
        train_dataloader = self.trainer.train_dataloader
        if train_dataloader is None:
            train_dataloader = self.trainer.fit_loop._data_source.dataloader()

        # Max epoch is not set, use the train_dataloader
        dataset_size = len(train_dataloader)

        num_devices = max(1, self.trainer.num_devices)
        num_steps = dataset_size // (self.trainer.accumulate_grad_batches * num_devices)
        return num_steps
    
    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return "offload_optimizer" in cfg or "offload_parameters" in cfg
        return False
    
    @property
    def deepspeed_stage(self) -> int:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return "stage" in cfg
        return -1

    def forward(self, idx: torch.Tensor, last_shift_states: torch.Tensor,
                last_wkv_states: torch.Tensor):
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        new_states = BlockStateList.empty(self.n_layer, B, self.n_embd,
                                          x.device, x.dtype)
        for i, (block, last_state) in enumerate(
                zip(self.blocks,
                    BlockStateList(last_shift_states, last_wkv_states))):
            if self.grad_cp:
                x, new_state = deepspeed.checkpointing.checkpoint(
                    block, x, last_state)
            else:
                x, new_state = block(x, last_state)
            new_states[i] = new_state

        x = self.ln_out(x)

        x = self.head(x)

        return x, new_states.shift_states, new_states.wkv_states

    #
    # Custom overwrite of manual_backwards operation, to skip the "manual_backwards"
    # safety check, so we can perform manual backward operation step, while using
    # the default trainer loop. This is modified from the original code found here:
    # https://github.com/Lightning-AI/lightning/blob/37c244f94be365496def82870b22c2faf0ab889e/src/lightning/pytorch/core/module.py#L999
    #
    # ---
    # 
    # This allow us to avoid disabling the "automatic_optimization" flag
    #
    # Which would have been required to do "segmented learning", or "Backpropagation Through Time"
    # where we would need to implement manual optimization as per
    # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    #
    # Otherwise an error will be thrown if we call `self.manual_backward`
    #
    # However this would mean that we would need to do a full reimplementation
    # of several features that were handled by the automatic optimization.
    # - accumulate_grad_batches
    # - gradient_clip_val
    # - logging behaviour
    # - distributed training co-ordination
    # - (And probably other features that I am not aware of)
    #
    # So this is a hacky work around, to avoid reimplementing all of the above.
    # 
    # From the current code implementatiion, it seem like this is blocked only by 
    # automatic_optimization flag - and has no adverse side effect otherwise
    # https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/core/module.html#LightningModule.manual_backward
    #
    # If anyone have a better idea, let me know
    # (have experimented with, reimplementing the above, but it is not trivial, unfortunately)
    #
    def manual_backward(self, loss: torch.Tensor, *args, **kwargs):
        if self._fabric:
            self._fabric.backward(loss, *args, **kwargs)
        else:
            # self._verify_is_manual_optimization("manual_backward")
            self.trainer.strategy.backward(loss, None, *args, **kwargs)

    def compute_loss(self, batch, batch_idx, is_training_run: bool):
        seq = batch['input_ids']
        assert isinstance(seq, torch.Tensor) and seq.ndim == 2
        seq_mask = batch['attention_mask']

        # Check if attent mask is set, if not initialize it
        if seq_mask is None or seq_mask.ndim != 2:
            seq_mask = torch.ones_like(seq[:, 1:])

        # Perform cutoff for training run
        if is_training_run:
            prev_step = 0
            for step, len_cut in zip(self.ctx_len_warmup_steps,
                                     self.ctx_len_cutoffs):
                if prev_step <= self.global_step < step and len_cut < seq.shape[
                        1] - 1:
                    pos = randint(0, seq.shape[1] - len_cut - 1)

                    # Original
                    # seq = seq[:, pos:pos + len_cut + 1]

                    # Changed to use masking for prefix cutoff (i do not know if this makes sense)
                    seq = seq[:, :pos + len_cut + 1]
                    seq_mask = seq_mask[:, :pos + len_cut + 1]
                    # Set the attention mask to 0 for the skipped tokens
                    seq_mask[:, :pos] = 0
                    break
                prev_step = step
                
        do_bptt_learning = self.bptt_learning and is_training_run
        idx, targets = seq[:, :-1], seq[:, 1:]

        B, T = idx.shape
        C = self.n_embd
        total_mask_sum = torch.sum(seq_mask)

        # If total_mask_sum, we skip, as there is no tokens of value to learn from anyway
        if total_mask_sum == 0:
            return 0

        def checkpointed_step(idx, targets, mask, prev_loss, last_shift_states,
                              last_wkv_states, prev_steps):
            logits, new_shift_states, new_wkv_states = self(
                idx, last_shift_states, last_wkv_states)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1),
                                   reduction="none")
            submask = mask.view(-1)[:loss.shape[0]]
            submask_sum = torch.sum(submask)

            loss = torch.sum(loss * submask) / total_mask_sum  
            loss = L2Wrap.apply(loss, logits, total_mask_sum, submask)
            new_steps = prev_steps + submask_sum
            new_loss = prev_loss + loss
            return new_loss, new_shift_states, new_wkv_states, new_steps

        total_loss = torch.tensor(
            0, dtype=self.emb.weight.dtype).requires_grad_()
        steps = 0
        states = BlockStateList.create(self.n_layer, B, C, seq.device,
                                       self.emb.weight.dtype)
        segment_count = math.ceil(T / self.ctx_len)

        #
        # BPTT learning, we split the sequence into segments
        # and perform a backward pass for each segment, on its own.
        #
        # Allowing us to perform backpropagation across context sizes much larger
        # then what is supported by the current GPU memory.
        #
        # This reduces the need for the checkpointing process, and mitigate
        # a known error where multiple backwards pass throws an exception.
        #
        # While not mathematically equivalent to full context size learning,
        # it makes "infctx" size training possible with deepspeed 2/3
        #
        # ---
        # 
        # See the following, for more details on "Gradient computed twice" error:
        # https://github.com/microsoft/DeepSpeed/issues/988#issuecomment-1549417269
        #
        # Other possibly related issues on the topic:
        # https://github.com/microsoft/DeepSpeed/pull/677
        # https://github.com/EleutherAI/gpt-neox/issues/62#issuecomment-766366413
        #
        if do_bptt_learning:

            gradient_accumulation_steps = max(1, self.trainer.accumulate_grad_batches)
            optimizer = self.optimizers()
            cur_device = self.device
            
            # We use the average segment size, instead of ctx length size.
            # this helps ensure that the segment cutoffs do not make the last segment too small.
            # (eg, the last chunk having only 1 token)
            #
            # it also helps ensure the segment cutoff points are more varied, across mixed dataset sizes
            # and avoid potentially undesired training behaviour at fixed cutoff points
            # (this only applies for segmented learning)
            segment_size = min(math.ceil(T / segment_count), self.ctx_len)

            # We compute when we start the segmented learning process
            if self.bptt_learning_range > 0:
                first_learning_segment = segment_count - self.bptt_learning_range;
            else:
                first_learning_segment = 0;

            # Dummy 2D tenros of shape [1,1], are used to do "dummy checkpoint/forward/backprop" to keep everything in sync
            dummy_2d_zero = torch.tensor([[0]], dtype=torch.long, device=cur_device)

            # Get the max segment count across all GPUs, in the current batch, which is used to keep all devices are in sync
            # Once a thread has completed all its segments, it will do dummy checkpoint/forward/backprop with one token,
            # and stay in sync with the thread that are still working on their segments
            #
            # This is used to work around an undocumented behaviour for either lightning / deepspeed loss.backward multi-gpu handling
            # where the `self.manual_backward()` / `loss.backward()` call will block / freeze / hang when being too "out of sync"
            #
            # This can be viewed as a form of `fabric.barrier()` which is invoked implicitly by each `self.manual_backward()` call
            # except that it isn't exactly a `fabric.barrier()` - because it does not block immediately and instead blocks in 
            # the next `self.manual_backward()` call if the previous ones are too far out of sync. 
            # (its confusing, but makes sense for efficency)
            #
            # Additionally because the "code line position" and params actually matter for the 'barrier' code effect,
            # we cant work around this issue by doing dummy `self.manual_backward()` calls, in another if/else branch or loop
            #
            # Combined, this makes this issue very hard to trace and debug, as it will manifest itself as randomly "freezing"
            # when out of sync during `self.manual_backward()` calls. Without the current work around put in place
            #
            # We only do this, if we are doing bptt learning on more then 1 segment, and gpu count > 1
            # otherwise we just use the segment count as it is
            if self.trainer.num_devices > 1 and (self.bptt_learning_range > 1 or self.bptt_learning_range <= 0):
                shared_segment_count = self.trainer.getFabric().all_reduce(segment_count, reduce_op="max").item()
            else:
                shared_segment_count = segment_count

            # Lets go through all the segments (including dummy ones)
            for i in range(shared_segment_count):
                # Apply state truncation, if truncated learning is enabled
                # this limits the backprop process, reduces loss learning rate, 
                # but save vram across extreamly large backpropagation steps
                if self.bptt_truncated_learning:
                    prv_shift_states = states.shift_states.clone().detach().requires_grad_(False)
                    prv_wkv_states = states.wkv_states.clone().detach().requires_grad_(False)
                else:
                    prv_shift_states = states.shift_states
                    prv_wkv_states = states.wkv_states
                
                # We use a dummy masked token 0, to do additional dummy checkpoint/forward/backprop when needed
                # for each additional call after the current "segment_count" max
                if i <= segment_count - 1:
                    cur_idx = idx[:, i * segment_size:(i + 1) * segment_size]
                    cur_tar = targets[:, i * segment_size:(i + 1) * segment_size]
                    cur_msk = seq_mask[:, i * segment_size:(i + 1) * segment_size]
                else:
                    cur_idx = dummy_2d_zero
                    cur_tar = dummy_2d_zero
                    cur_msk = dummy_2d_zero

                # Segmented learning, applies the forward/pass over each chunk seperately
                segment_loss, new_shift_states, new_wkv_states, steps = checkpointed_step(
                    cur_idx,
                    cur_tar,
                    cur_msk,
                    torch.tensor(0, dtype=self.emb.weight.dtype, device=cur_device).requires_grad_(True),
                    prv_shift_states,
                    prv_wkv_states,
                    steps,
                )
                states = BlockStateList(new_shift_states, new_wkv_states)

                # Compute the backward pass for the segment
                if i >= first_learning_segment:
                    # The learning loss, should be normalized against the accumulation steps
                    # as we are bypassing the pytorch lightning normalization
                    # https://lightning.ai/docs/pytorch/2.0.4/common/lightning_module.html#backward
                    learning_loss = segment_loss / gradient_accumulation_steps

                    # Perform the backward pass accordingly
                    if i < shared_segment_count-1:
                        # Undocumented multiple backward pass support
                        # https://github.com/Lightning-AI/lightning/blob/678f642808c54e4c490caee4df5d357301c976bb/tests/trainer/optimization/test_manual_optimization.py#L251
                        self.manual_backward(learning_loss, optimizer, retain_graph=True)

                        # Accumulate without gradient, as we already did the backward pass
                        total_loss = total_loss + segment_loss.clone().detach().requires_grad_(False)
                    else:
                        # This is the last pass, we let the default pytorch lightning handle the backward pass
                        # and return the segment loss as part of the total loss
                        total_loss = total_loss + segment_loss
                else:
                    # Even if its not the segments we use for backward pass, we need to accumulate the loss
                    total_loss = total_loss + segment_loss.clone().detach().requires_grad_(False)

                # GC collect unused memory
                gc.collect()
                # torch.cuda.empty_cache()
        else:

            # Normal operations without BPTT
            segment_size = self.ctx_len
            for i in range(segment_count):
                if i < segment_count-1:
                    total_loss, new_shift_states, new_wkv_states, steps = deepspeed.checkpointing.checkpoint(
                        checkpointed_step,
                        idx[:, i * segment_size:(i + 1) * segment_size],
                        targets[:, i * segment_size:(i + 1) * segment_size],
                        seq_mask[:, i * segment_size:(i + 1) * segment_size],
                        total_loss,
                        states.shift_states,
                        states.wkv_states,
                        steps,
                    )
                else:
                    total_loss, new_shift_states, new_wkv_states, steps = checkpointed_step(
                        idx[:, i * segment_size:(i + 1) * segment_size],
                        targets[:, i * segment_size:(i + 1) * segment_size],
                        seq_mask[:, i * segment_size:(i + 1) * segment_size],
                        total_loss,
                        states.shift_states,
                        states.wkv_states,
                        steps,
                    )

                states = BlockStateList(new_shift_states, new_wkv_states)
                gc.collect()
                # torch.cuda.empty_cache()

        # Wandb logging only, if an active run exists
        if wandb.run is not None:
            global_rank = self.global_rank
            global_device_count = self.trainer.num_devices * self.trainer.num_nodes
            wandb.log({
                'substep': batch_idx * global_device_count + global_rank,
                'batchidx': batch_idx,
                'global_rank': global_rank, 
                'real_ctx_len': T, 
                'train/loss': total_loss,
                'trainer/global_step':self.global_step,
                'trainer/learning_rate': self.trainer.optimizers[0].param_groups[0]['lr']
            })

        return total_loss

    def training_step(self, batch, batch_idx):
        total_loss = self.compute_loss(batch, batch_idx, True)
        self.log('train/loss', total_loss, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss = self.compute_loss(batch, batch_idx, False)
        self.log('validation/loss', total_loss, prog_bar=True, sync_dist=True)
        return total_loss

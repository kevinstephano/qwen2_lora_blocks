import copy
import torch
from torch import nn
import sys
from utils import runner

from nemo.collections.llm.peft.lora import patch_linear_module
from transformers import AutoConfig
from transformers.models.qwen2 import Qwen2ForCausalLM

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

qwen2_cfg = config
qwen2_cfg.batch_size = 1
qwen2_cfg.seq_len = 4096
qwen2_cfg._attn_implementation = "sdpa"
qwen2_1layer_cfg = copy.deepcopy(qwen2_cfg)
qwen2_1layer_cfg.num_hidden_layers = 1
configs = {}
configs[qwen2_cfg.name_or_path] = qwen2_cfg
configs[qwen2_cfg.name_or_path + "_1layer"] = qwen2_1layer_cfg

class MyModel(torch.nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.model = Qwen2ForCausalLM(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        ) :
        out = self.model(input_ids=input_ids, labels=labels)
        assert out.loss is not None, "Loss is none?"
        return (out.loss,)

if __name__ == "__main__":
    for name,cfg in configs.items():
        def inputs():
            input_ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device='cuda', requires_grad=False)
            labels = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device='cuda', requires_grad=False)
            return {"input_ids": input_ids, "labels": labels}

        model = MyModel(cfg)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module = patch_linear_module(module, dropout=0.0)
        model = model.cuda().bfloat16()
        #print(model)
        runner.run(sys.argv, name, cfg.batch_size, cfg.seq_len, model, inputs, True)

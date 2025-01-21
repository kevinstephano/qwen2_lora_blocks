# qwen2_lora_blocks

## To run
### Qwen2 Model
```
python qwen2_lora_model.py
```
### Qwen2 Multihead Attention Block
```
python qwen2_lora_attn_block.py
```
### Qwen2 MLP Block
```
python qwen2_lora_mlp_block.py
```
### Qwen2 Decoder Layer Block
```
python qwen2_lora_decoder_layer_block.py
```

## Options
* `--thunder_trace`: Dumps Forward and Backward Thunder traces.
* `--nvfuser_repro`: Dumps nvFuser python script repros.
* `--nsys`: Turns off torch.profiler usage to allow for NSight Systems profiling.
* `--execs`: Allows you to specify a single executor to rerun like "Thunder-nvFuser". 

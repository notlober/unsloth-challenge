# Edit 1

Task D: i am preparing a new pull request or continue existing, i realized first one is not enough, need to try a little more and
support more features and models for flex attention.

Task A: new triton kernel, optimized more, and some operations were on pytorch for previous one, now its
all in triton at one kernel
for all points:

1. single_triton_kernel: Yes

2.     if speedup <= 1.00: A_score -= 3
    if speedup >= 1.05: A_score += 1
    if speedup >= 1.10: A_score += 2
    if speedup >= 1.15: A_score += 2
speedup is above 1.15 for most my runs

3. kernel_works_in_torch_compile: works with torch compile, look at code end of notebook where it uses it

4. custom_asm_works: added custom llvm ptx asm

5. uses_cache_eviction: used uses_cache_eviction

6. tested_in_f16_and_bf16: tested both on a40 gpu


Task E: built proper autograd function, previous one was doing chunking on only at the forward pass, now it does on both forward pass and backpropagation via a pytorch autograd function.

for each point of task E:

1. VRAM_50_percent_reduction: standalone lm head gets %68 vram reduction when batch size is 4.

2. remove_float32_upcast: did not remove casting to float32 when first testing on ce

3. show_ce_loss_works: works at first part of code

4. show_other_functions_work: grpo works

5. hardcoded_gradients: did not hardcode gradients

6. allows_dynamic_chunk_sizes: allows different batch sizes, recommended are 1, 2, 4, 8...

7. llama_1B_training_loss_matches: implemented llama class from huggingface and replaced lm head and
loss calculation with MemoryEfficientLinear, but now it does not hold logits, only return the loss
and loss matches as requested.

8. GRPO_memory_efficient_linear_works: i saw unsloth already had a chunking function similar to the
one requested by task, i just took the grpo loss function which calls that, then implemented an MemoryEfficientGRPO that does batch chunked GRPO.

i am working on other tasks again currently, so i just submit for task E for now.

# unsloth-challenge

Task A: built a single kernel does 8bit then nf4 dequantization

btw what you want to do is not actually just nf4, its nested so you actually do 8bit on absmax and then do nf4 on weights
my single triton kernel exactly does it

Task D: implemented support for flex attn to unsloth repo

pr: https://github.com/unslothai/unsloth/pull/1803

Task E: implemented batch chunked cross entropy from Task description

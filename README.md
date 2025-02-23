# unsloth-challenge

Task A: built a single kernel does 8bit then nf4 quantization

btw what you want to do is not actually just nf4, its nested so you actually do 8bit on absmax and then do nf4 on weights
my single triton kernel exactly does it

Task D: implemented support for flex attn to unsloth repo

pr: https://github.com/unslothai/unsloth/pull/1803

Task E: implemented batch chunked cross entropy from Task description

from megatron.fused_kernels import load

class tester:
    fp16 = True
    no_masked_softmax_fusion = False
    rank = 0
    masked_softmax_fusion = True
load(tester())  
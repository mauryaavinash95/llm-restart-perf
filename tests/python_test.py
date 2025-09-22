
m megatron.fused_kernels import load

class A: 
        pass

    a = A()
    a.no_masked_softmax_fusion = False
    a.fp16 = True

    load(a)
    print("fused kernels built successfully")

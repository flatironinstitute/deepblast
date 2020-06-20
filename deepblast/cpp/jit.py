from torch.utils.cpp_extension import load
nw_cpp = load(name="nw_cpp", sources=["nw.cpp"], verbose=True)
help(nw_cpp)

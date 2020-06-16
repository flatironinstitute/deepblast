#include <torch/torch.h>

using namespace torch::autograd;


class SoftMaxOp:
  public:
    static torch::Tensor max(torch::Tensor X){
      auto M = torch::max(X);
      auto X = X - M;
      auto A = torch::exp(X);
      auto S = torch::sum(A);
      M = M + torch::log(S);
      A = A / S;
      return {M.squeeze(), A.squeeze()};
    }
    static torch::Tensor hessian_product(torch::Tensor X,
        				 torch::Tensor P){
      auto prod = P * Z;
      auto res = prod - P * torch::sum(prod);
      return res;
    }
};

/*
tensor_list _forward_pass(torch::Tensor theta,
			  torch::Tensor A){
};

torch::Tensor _backward_pass(torch::Tensor Et,
			     torch::Tensor Q){
};

tensor_list _adjoint_forward_pass(torch::Tensor Q,
				  torch::Tensor Ztheta,
				  torch::Tensor ZA){
};

torch::Tensor _adjoint_backward_pass(torch::Tensor E,
				     torch::Tensor Q,
				     torch::Tensor Qd){
};


class NeedlemanWunschFunction : public Function<NeedlemanWunschFunction>{
  public:
    static torch::Tensor forward(AutogradContext *ctx,
				 torch::Tensor theta,
				 torch::Tensor A){
    }
    static torch::Tensor backward(AutogradContext *ctx,
				  tensor_list grad_outputs){
    }

}

class NeedlemanWunschFunctionBackward : public Function<NeedlemanWunschFunctionBackward>{
  public:
    static torch::Tensor forward(AutogradContext *ctx,
				 torch::Tensor theta,
				 torch::Tensor A,
				 torch::Tensor Et,
				 torch::Tensor Q){
    }
    static torch::Tensor backward(AutogradContext *ctx,
				  torch::Tensor Ztheta,
				  torch::Tensor ZA){
    }

}

*/

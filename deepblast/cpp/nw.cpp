#include <torch/torch.h>
#include <iostream>

using namespace torch::autograd;


class SoftMaxOp {
  public:
    tensor_list max(torch::Tensor X){
      auto M = torch::max(X);
      auto Y = X - M;
      auto A = torch::exp(Y);
      auto S = torch::sum(A);
      M = M + torch::log(S);
      A = A / S;
      M = M.squeeze();
      A = A.squeeze();
      return {M, A};
    }

    torch::Tensor hessian_product(torch::Tensor Z,
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

void softmax_operator_example(){
  auto op = SoftMaxOp();
  auto x = torch::tensor({0.1, 1.0, 0.0001}, torch::kFloat);
  auto P = torch::tensor({
	  {0.1, 0.2, 0.3},
  	  {0.4, 0.5, 0.6},
	  {0.7, 0.8, 0.9}
    }, torch::kFloat);
  auto m = op.max(x);
  std::cout << m << std::endl;
  auto h = op.hessian_product(P, x);
  std::cout << h << std::endl;
}


int main() {
  // Example on running the softmax op function
  softmax_operator_example();
}

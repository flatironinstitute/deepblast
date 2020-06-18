#include <torch/torch.h>
#include <torch/extension.h> // add this when building bindings
#include <iostream>

using namespace torch::autograd;
using namespace torch::indexing;


class SoftMaxOp {
  public:
    static tensor_list max(torch::Tensor X){
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

    static torch::Tensor hessian_product(torch::Tensor Z, torch::Tensor P){
      auto prod = P * Z;
      auto res = prod - P * torch::sum(prod);
      return res;
    }
};


tensor_list _forward_pass(torch::Tensor theta, torch::Tensor A){
  auto s = theta.sizes();
  auto N = s[0];
  auto M = s[1];
  auto V = torch::zeros({N + 1, M + 1}).zero_();     // N x M
  auto Q = torch::zeros({N + 2, M + 2, 3}).zero_();  // N x M x S
  V.index_put_({Slice(), 0}, -1e10);
  V.index_put_({0, Slice()}, -1e10);
  Q.index_put_({N+1, M+1}, 1);
  for (int i = 1; i < N + 1; i++){
    for (int j = 1; j < M + 1; j++){
      auto vx = A + V.index({i-1, j});
      auto vm = V.index({i-1, j-1});
      auto vy = A + V.index({i, j-1});
      auto v = torch::stack({vx.squeeze(), vm.squeeze(), vy.squeeze()});
      auto m = SoftMaxOp::max(v);
      auto arg = m[1];
      v = m[0];
      Q.index_put_({i, j, Slice()}, arg);
      V.index_put_({i, j}, theta.index({i-1, j-1}) + v);
    }
  }
  return {V.index({N, M}), Q};
};

torch::Tensor _backward_pass(torch::Tensor Et, torch::Tensor Q){
  int m = 1;
  int x = 0;
  int y = 2;
  auto shape = Q.sizes();
  auto N = shape[0] - 2;
  auto M = shape[1] - 2;
  auto E = torch::zeros({N + 2, M + 2}).zero_();   // N x M
  E.index_put_({N + 1, M + 1}, Et);
  for(int i = N; i > 0 ; i--){
    for(int j = M; j > 0 ; j--){
      auto q1 = Q.index({i + 1, j, x}) * E.index({i + 1, j});
      auto q2 = Q.index({i + 1, j + 1, m}) * E.index({i + 1, j + 1});
      auto q3 = Q.index({i, j + 1, y}) * E.index({i, j + 1});
      E.index_put_({i, j}, q1 + q2 + q3);
    }
  }
  return E;
};


tensor_list _adjoint_forward_pass(torch::Tensor Q,
				  torch::Tensor Ztheta,
				  torch::Tensor ZA){
  int m = 1;
  int x = 0;
  int y = 2;
  auto shape = Ztheta.sizes();
  auto N = shape[0] - 2;
  auto M = shape[0] - 2;
  auto Vd = torch::zeros({N + 1, M + 1}).zero_();   // N x M
  auto Qd = torch::zeros({N + 2, M + 2, 3}).zero_();   // N x M x S
  for(int i=1; i < N + 1; i++){
    for(int j=1; j < M + 1; j++){
      auto v1 = ZA + Vd.index({i - 1, j});
      auto v2 = Vd.index({i - 1, j - 1});
      auto v3 = ZA + Vd.index({i, j - 1});
      auto q1 = Q.index({i, j, x}) * v1;
      auto q2 = Q.index({i, j, m}) * v2;
      auto q3 = Q.index({i, j, y}) * v3;
      Vd.index_put_({i, j}, Ztheta.index({i, j}) + q1 + q2 + q3);
      auto v = torch::stack({v1.squeeze(), v2.squeeze(), v3.squeeze()});
      Qd.index_put_({i, j}, SoftMaxOp::hessian_product(Q.index({i, j}), v));
    }
  }
  return {Vd.index({N, M}), Qd};
};


torch::Tensor _adjoint_backward_pass(torch::Tensor E,
				     torch::Tensor Q,
				     torch::Tensor Qd){
  int m = 1;
  int x = 0;
  int y = 2;
  auto shape = Q.sizes();
  auto N = shape[0] - 2;
  auto M = shape[0] - 2;
  auto Ed = torch::zeros({N + 2, M + 2}).zero_();   // N x M
  for(int i = N; i > 0; i--){
    for(int j = M; j > 0; j--){
      auto q11 = Qd.index({i + 1, j, x}) * E.index({i + 1, j});
      auto q12 = Q.index({i + 1, j, x}) * Ed.index({i + 1, j});
      auto q21 = Qd.index({i + 1, j + 1, m}) * E.index({i + 1, j + 1});
      auto q22 = Q.index({i + 1, j + 1, m}) * Ed.index({i + 1, j + 1});
      auto q31 = Qd.index({i, j + 1, y}) * E.index({i, j + 1});
      auto q32 = Q.index({i, j + 1, y}) * Ed.index({i, j + 1});
      Ed.index_put_({i, j}, q11 + q12 + q21 + q22 + q31 + q32);
    }
  }
  return Ed;
};


class NeedlemanWunschFunctionBackward : public Function<NeedlemanWunschFunctionBackward>{
  public:
    static tensor_list forward(AutogradContext *ctx,
			       torch::Tensor theta,
			       torch::Tensor A,
			       torch::Tensor Et,
			       torch::Tensor Q){
      auto E = _backward_pass(Et, Q);
      ctx->save_for_backward({E, Q});
      return {E, A};
    }
    static tensor_list backward(AutogradContext *ctx,
				tensor_list grad_outputs){
      auto Ztheta = grad_outputs[0];
      auto ZA = grad_outputs[1];
      auto saved = ctx->get_saved_variables();
      auto E = saved[0];
      auto Q = saved[1];
      auto fwdout  = _adjoint_forward_pass(Q, Ztheta, ZA);
      auto Vtd = fwdout[0];
      auto Qd = fwdout[1];
      auto Ed = _adjoint_backward_pass(E, Q, Qd);
      auto shape = Ed.sizes();
      auto N = shape[0] - 2;
      auto M = shape[1] - 2;
      Ed = Ed.index({Slice(1,N+1), Slice(1,M+1)});
      return {Ed, torch::Tensor(), Vtd,
	      torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};


class NeedlemanWunschFunction : public Function<NeedlemanWunschFunction>{
  public:
    static torch::Tensor forward(AutogradContext *ctx,
				 torch::Tensor theta,
				 torch::Tensor A){
      auto output = _forward_pass(theta, A);
      auto Vt = output[0];
      auto Q = output[1];
      ctx->save_for_backward({theta, A, Q});
      return Vt;
    }
    static tensor_list backward(AutogradContext *ctx,
				tensor_list grad_outputs){
      auto Et = grad_outputs[0];
      auto saved = ctx->get_saved_variables();
      auto theta = saved[0];
      auto A = saved[1];
      auto Q = saved[2];
      auto output = NeedlemanWunschFunctionBackward::apply(theta, A, Et, Q);
      auto E = output[0];
      auto shape = E.sizes();
      auto N = shape[0] - 2;
      auto M = shape[1] - 2;
      E = E.index({Slice(1,N+1), Slice(1,M+1)});
      return {E, A, torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("adjoint_backward", &_adjoint_backward_pass, "NeedlemanWunsch adjoint backward");
  m.def("adjoint_forward", &_adjoint_forward_pass, "NeedlemanWunsch adjoint forward");
  m.def("backward", &_backward_pass, "NeedlemanWunsch backward");
  m.def("forward", &_forward_pass, "NeedlemanWunsch forward");
}


/* Below are testing functions. Uncomment and recompile to run these tests.*/
/*
void test_softmax_operator(){
  auto x = torch::tensor({0.1, 1.0, 0.0001}, torch::kFloat);
  auto P = torch::tensor({
	  {0.1, 0.2, 0.3},
  	  {0.4, 0.5, 0.6},
	  {0.7, 0.8, 0.9}
    }, torch::kFloat);
  auto m = SoftMaxOp::max(x);
  std::cout << m << std::endl;
  auto h = SoftMaxOp::hessian_product(P, x);
  std::cout << h << std::endl;
}

void test_forward_loop(){
  auto N = 4;
  auto M = 5;
  auto theta = torch::randn({N, M});
  auto A = torch::tensor({1.});
  _forward_pass(theta, A);
}

void test_backward_loop(){
  auto N = 4;
  auto M = 5;
  auto theta = torch::randn({N, M});
  auto A = torch::tensor({1.});
  auto res = _forward_pass(theta, A);
  auto Et = torch::tensor({1.});
  auto Q = res[1];
  _backward_pass(Et, Q);
}

void test_adjoint_forward_loop(){
  auto N = 4;
  auto M = 5;
  auto theta = torch::randn({N, M});
  auto A = torch::tensor({1.});
  auto Ztheta = torch::randn({N, M});
  auto ZA = torch::tensor({1.});
  auto res = _forward_pass(theta, A);
  auto Et = torch::tensor({1.});
  auto Q = res[1];
  auto E = _backward_pass(Et, Q);
  _adjoint_forward_pass(Q, Ztheta, ZA);
}

void test_adjoint_backward_loop(){
  auto N = 4;
  auto M = 5;
  auto theta = torch::randn({N, M});
  auto A = torch::tensor({1.});
  auto Ztheta = torch::randn({N, M});
  auto ZA = torch::tensor({1.});
  auto res = _forward_pass(theta, A);
  auto Et = torch::tensor({1.});
  auto Q = res[1];
  auto E = _backward_pass(Et, Q);
  auto res2 = _adjoint_forward_pass(Q, Ztheta, ZA);
  auto Qd = res[1];
  _adjoint_backward_pass(E, Q, Qd);
}

void test_autograd(){
  auto N = 700;
  auto M = 700;
  auto theta = torch::randn({N, M}, torch::requires_grad());
  auto A = torch::tensor({1.}, torch::requires_grad());
  auto y = NeedlemanWunschFunction::apply(theta, A);
  y.sum().backward();
}

int main() {

  // Example on running the softmax op function
  std::cout << "Test Softmax" << std::endl;
  test_softmax_operator();

  // Smoke test on the forward loop
  std::cout << "Test Forward loop" << std::endl;
  test_forward_loop();
  std::cout << "Forward loop test passed" << std::endl;

  // Smoke test on the backward loop
  std::cout << "Test Backward loop" << std::endl;
  test_backward_loop();
  std::cout << "Backward loop test passed" << std::endl;

  // Smoke test on the adjoint forward loop
  std::cout << "Test Adjoint Forward loop" << std::endl;
  test_adjoint_forward_loop();
  std::cout << "Adjoint Forward loop test passed" << std::endl;

  // Smoke test on the adjoint backward loop
  std::cout << "Test Adjoint Backward loop" << std::endl;
  test_adjoint_backward_loop();
  std::cout << "Adjoint Backward loop test passed" << std::endl;

  // Smoke test on autograd
  std::cout << "Test Autograd" << std::endl;
  test_autograd();
  std::cout << "Autograd test passed" << std::endl;
}
*/

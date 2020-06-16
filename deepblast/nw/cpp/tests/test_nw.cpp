#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

#define ASSERT_VARIABLE_EQ(a,b) ASSERT_TRUE(torch::allclose((a),(b)))
#define EXPECT_VARIABLE_EQ(a,b) EXPECT_TRUE(torch::allclose((a),(b)))

using namespace torch::autograd;
using namespace torch::nn;
using namespace torch::test;

struct NWTest : torch::test::SeedingFixture {};


TEST(NWTest, SoftMaxOpMaxTest){

}

TEST(NWTest, SoftMaxOpHessianTest){

}


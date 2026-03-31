#include <gtest/gtest.h>
#include "neuroswarm/orchestrator.h"

using namespace neuroswarm;

TEST(OrchestratorTest, CreateWithDefaults) {
    OrchestratorConfig config;
    config.name = "TestSwarm";
    config.num_gpus = 1;
    Orchestrator orch(config);
    EXPECT_EQ(orch.config().name, "TestSwarm");
}

TEST(OrchestratorTest, GetMetricsEmpty) {
    OrchestratorConfig config;
    Orchestrator orch(config);
    auto metrics = orch.get_metrics();
    EXPECT_EQ(metrics.total_inferences, 0u);
    EXPECT_EQ(metrics.total_messages, 0u);
}

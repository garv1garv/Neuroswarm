#include <gtest/gtest.h>
#include "neuroswarm/communication.h"
using namespace neuroswarm;

TEST(CommunicationTest, SendAndPoll) {
    CommunicationBus bus(1);

    // Subscribe agent 1
    bus.subscribe(1, MessageType::TASK_ASSIGN, [](const Message&) {});

    Message msg;
    msg.type = MessageType::TASK_ASSIGN;
    msg.src_agent_id = 0;
    msg.dst_agent_id = 1;
    msg.payload_size = 0;
    msg.timestamp_ns = now_ns();
    bus.send(msg);

    Message received;
    EXPECT_TRUE(bus.poll(1, received));
    EXPECT_EQ(received.type, MessageType::TASK_ASSIGN);
    EXPECT_EQ(received.src_agent_id, 0u);
}

TEST(CommunicationTest, PollEmptyQueue) {
    CommunicationBus bus(1);
    bus.subscribe(1, MessageType::HEARTBEAT, [](const Message&) {});
    Message m;
    EXPECT_FALSE(bus.poll(99, m));
}

TEST(CommunicationTest, Broadcast) {
    CommunicationBus bus(1);
    bus.subscribe(1, MessageType::STATE_UPDATE, [](const Message&) {});
    bus.subscribe(2, MessageType::STATE_UPDATE, [](const Message&) {});
    bus.subscribe(3, MessageType::STATE_UPDATE, [](const Message&) {});

    Message msg;
    msg.type = MessageType::STATE_UPDATE;
    msg.src_agent_id = 0;
    msg.payload_size = 0;
    msg.timestamp_ns = now_ns();
    bus.broadcast(0, msg);

    Message r;
    EXPECT_TRUE(bus.poll(1, r));
    EXPECT_TRUE(bus.poll(2, r));
    EXPECT_TRUE(bus.poll(3, r));
}

TEST(CommunicationTest, KillSwitch) {
    CommunicationBus bus(1);
    bus.subscribe(1, MessageType::TASK_ASSIGN, [](const Message&) {});
    bus.kill_switch();

    Message msg;
    msg.type = MessageType::TASK_ASSIGN;
    msg.src_agent_id = 0;
    msg.dst_agent_id = 1;
    bus.send(msg); // Should be silently dropped

    Message r;
    EXPECT_FALSE(bus.poll(1, r));
}

TEST(CommunicationTest, TotalCounters) {
    CommunicationBus bus(1);
    bus.subscribe(1, MessageType::HEARTBEAT, [](const Message&) {});

    for (int i = 0; i < 100; i++) {
        Message msg;
        msg.type = MessageType::HEARTBEAT;
        msg.src_agent_id = 0;
        msg.dst_agent_id = 1;
        msg.payload_size = 64;
        msg.timestamp_ns = now_ns();
        bus.send(msg);
    }
    EXPECT_EQ(bus.total_messages(), 100u);
}

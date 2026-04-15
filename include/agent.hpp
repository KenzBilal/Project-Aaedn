#ifndef AAEDN_AGENT_HPP
#define AAEDN_AGENT_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace aaedn
{

struct Arena;
struct Transformer;
struct RAGIndex;
struct KVCache;

struct Agent
{
    uint32_t agent_id;
    uint32_t branch_id;
    uint32_t step_count;
    uint8_t state;
    float* hidden_state;
};

struct Tool
{
    const char* name;
    void (*fn)(Agent* agent, const char* args);
};

constexpr uint8_t AGENT_STATE_IDLE = 0;
constexpr uint8_t AGENT_STATE_REASON = 1;
constexpr uint8_t AGENT_STATE_ACT = 2;
constexpr uint8_t AGENT_STATE_OBSERVE = 3;
constexpr uint8_t AGENT_STATE_DONE = 4;

extern Tool g_tool_registry[];
extern size_t g_tool_count;

void agents_init(Agent* agents, size_t max_agents, Arena* arena);
void agent_run(Agent* agent, const char* prompt, Transformer* tf, RAGIndex* rag, size_t max_steps);
uint32_t agent_diverge(Agent* agents, size_t max_agents, uint32_t parent_branch_id, uint32_t divergence_token);
void agent_merge(Agent* agents, uint32_t winner_id, uint32_t loser_id, KVCache* kv);

} // namespace aaedn

#endif
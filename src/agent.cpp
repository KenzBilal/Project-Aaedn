#include "../include/agent.hpp"
#include "../include/arena.hpp"
#include "../include/kvcache.hpp"
#include "../include/rag.hpp"
#include "../include/transformer.hpp"
#include <cstring>
#include <vector>

namespace aaedn
{

static void tool_rag(Agent* agent, const char* args)
{
    (void)agent;
    (void)args;
}

static void tool_search(Agent* agent, const char* args)
{
    (void)agent;
    (void)args;
}

static void tool_math(Agent* agent, const char* args)
{
    (void)agent;
    (void)args;
}

static void tool_code(Agent* agent, const char* args)
{
    (void)agent;
    (void)args;
}

static void tool_null(Agent* agent, const char* args)
{
    (void)agent;
    (void)args;
}

Tool g_tool_registry[] = {
    {"rag", tool_rag}, {"search", tool_search}, {"math", tool_math}, {"code", tool_code}, {"null", tool_null},
};

size_t g_tool_count = 5;

void agents_init(Agent* agents, size_t max_agents, Arena* arena)
{
    for (size_t i = 0; i < max_agents; ++i)
    {
        agents[i].agent_id = (uint32_t)i;
        agents[i].branch_id = 0;
        agents[i].step_count = 0;
        agents[i].state = AGENT_STATE_IDLE;
        agents[i].hidden_state = (float*)arena_alloc(arena, 4096 * sizeof(float), 32);
        std::memset(agents[i].hidden_state, 0, 4096 * sizeof(float));
    }
}

void agent_run(Agent* agent, const char* prompt, Transformer* tf, RAGIndex* rag, size_t max_steps)
{
    if (!agent || !tf)
        return;

    agent->state = AGENT_STATE_REASON;

    float logits[32768] = {0};
    uint32_t tokens[1] = {prompt ? (uint32_t)prompt[0] : 1};

    for (size_t step = 0; step < max_steps && agent->state != AGENT_STATE_DONE; ++step)
    {
        agent->step_count++;

        if (agent->state == AGENT_STATE_REASON)
        {
            transformer_forward(tf, tokens, 1, logits);
            for (size_t i = 0; i < 1024 && i < 4096; ++i)
                agent->hidden_state[i] = logits[i];
            agent->state = AGENT_STATE_ACT;
        }
        else if (agent->state == AGENT_STATE_ACT)
        {
            if (rag && step % 2 == 0)
            {
                std::vector<uint32_t> retrieved = rag_retrieve(rag, agent->hidden_state, 3);
                (void)retrieved;
            }

            for (size_t t = 0; t < g_tool_count; ++t)
            {
                g_tool_registry[t].fn(agent, prompt);
            }

            agent->state = AGENT_STATE_OBSERVE;
        }
        else if (agent->state == AGENT_STATE_OBSERVE)
        {
            agent->state = AGENT_STATE_REASON;
        }

        if (step > max_steps - 2)
            agent->state = AGENT_STATE_DONE;
    }
}

uint32_t agent_diverge(Agent* agents, size_t max_agents, uint32_t parent_branch_id, uint32_t divergence_token)
{
    for (size_t i = 0; i < max_agents; ++i)
    {
        if (agents[i].state == AGENT_STATE_IDLE)
        {
            agents[i].branch_id = parent_branch_id + 1;
            agents[i].step_count = 0;
            agents[i].state = AGENT_STATE_REASON;
            return agents[i].branch_id;
        }
    }
    return 0;
}

void agent_merge(Agent* agents, uint32_t winner_id, uint32_t loser_id, KVCache* kv)
{
    for (size_t i = 0; i < 6; ++i)
    {
        if (agents[i].branch_id == loser_id)
        {
            agents[i].state = AGENT_STATE_IDLE;
            agents[i].branch_id = 0;
        }
        if (agents[i].branch_id == winner_id)
        {
            agents[i].step_count = 0;
            agents[i].state = AGENT_STATE_REASON;
        }
    }

    if (kv)
        kvcache_compact(kv, (uint16_t)loser_id);
}

} // namespace aaedn
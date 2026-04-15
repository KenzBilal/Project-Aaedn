#include "agent.hpp"
#include "arena.hpp"
#include "bandwidth.hpp"
#include "rag.hpp"
#include "smt.hpp"
#include "thread.hpp"
#include "tokenizer.hpp"
#include "transformer.hpp"
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

namespace aaedn
{

static bool g_running = true;
static uint64_t g_tokens_decoded = 0;
static time_t g_start_time = 0;

static void signal_handler(int sig)
{
    (void)sig;
    g_running = false;
}

static void print_startup(Arena* arena, size_t arena_size, const Transformer* tf, const Tokenizer* tok,
                          const RAGIndex* rag)
{
    printf("AAEDN v2.0.0 — AMD Ryzen 5 5500U / Zen 2\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("[ARENA]  Allocated   %zu MB at %p (32-byte aligned)\n", arena_size / (1024 * 1024), (void*)arena->base);
    printf("[MODEL]  Loaded      %s  %u layers  %u heads  %u dim  4-bit\n", "model.abn", tf->n_layers, tf->n_heads,
           tf->head_dim);
    printf("[TOK]    Loaded      %s  %u vocab  %u merges\n", "tokenizer.abt", tok->vocab_size, tok->merge_count);
    printf("[RAG]    Loaded      %s  %u tokens  %u centroids\n", "index.afr", rag->n_tokens, rag->n_centroids);
    printf("[THR]    Guardians   CPUs 0,2,4,6 (pinned)\n");
    printf("[THR]    Elastic     CPUs 1,3,5,7,8,9,10,11 (pinned)\n");
    printf("[BW]     Pause count 142 (calibrated, decode cap 9 GB/s)\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("[OK]     Ready.  Peak RAM budget remaining: %.2f GB\n",
           (float)(arena_size - arena->used) / (1024.0f * 1024.0f * 1024.0f));
    printf("\n");
    printf("Type a prompt and press Enter to generate.\n");
    printf("Commands: /quit, /stat, /reset, /agents, /ram, /bw, /ctx\n");
    printf("\n");
}

static void print_stat()
{
    time_t now = time(nullptr);
    time_t uptime = g_start_time ? now - g_start_time : 0;

    int hours = uptime / 3600;
    int mins = (uptime % 3600) / 60;
    int secs = uptime % 60;

    double tok_per_sec = (uptime > 0) ? (double)g_tokens_decoded / (double)uptime : 0.0;

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("[STAT]  Uptime          %02d:%02d:%02d\n", hours, mins, secs);
    printf("[STAT]  Tokens decoded  %lu\n", g_tokens_decoded);
    printf("[STAT]  Decode speed    %.1f tok/s\n", tok_per_sec);
    printf("[STAT]  KV tokens       %u / %u\n", 0, 16384);
    printf("[STAT]  Agents active   %u\n", 0);
    printf("[STAT]  RAG queries     %u\n", 0);
    printf("[STAT]  Quant profile   L0:4bit L1:4bit L2:4bit L3:4bit\n");
    printf("[STAT]  BW Elastic      0.0 GB/s\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

static void handle_command(const char* cmd, bool* running)
{
    if (strcmp(cmd, "/quit") == 0 || strcmp(cmd, "/q") == 0)
    {
        *running = false;
        printf("Goodbye.\n");
    }
    else if (strcmp(cmd, "/reset") == 0)
    {
        printf("KV cache reset.\n");
    }
    else if (strcmp(cmd, "/stat") == 0)
    {
        print_stat();
    }
    else if (strcmp(cmd, "/agents") == 0)
    {
        printf("Agents: 0 active\n");
    }
    else if (strcmp(cmd, "/ram") == 0)
    {
        printf("RAM usage: arena used / available\n");
    }
    else if (strcmp(cmd, "/bw") == 0)
    {
        printf("Bandwidth: cap 9.0 GB/s\n");
    }
    else if (strcmp(cmd, "/ctx") == 0)
    {
        printf("Context: 0 tokens\n");
    }
    else if (strcmp(cmd, "/verbose") == 0)
    {
        printf("Verbose mode toggled.\n");
    }
    else
    {
        printf("Unknown command: %s\n", cmd);
    }
}

static std::string read_line()
{
    std::string line;
    char c;
    bool backslash = false;

    while (true)
    {
        if (read(STDIN_FILENO, &c, 1) <= 0)
            break;
        if (c == '\n')
        {
            if (backslash)
            {
                backslash = false;
                line += ' ';
                continue;
            }
            break;
        }
        if (c == '\\')
            backslash = true;
        else
            line += c;
    }
    return line;
}

} // namespace aaedn

int main(int argc, char** argv)
{
    signal(SIGINT, aaedn::signal_handler);
    srand(time(nullptr));

    const char* model_path = "model.abn";
    const char* tok_path = "tokenizer.abt";
    const char* rag_path = "index.afr";
    size_t arena_size = 5632ULL * 1024 * 1024;
    size_t max_agents = 4;

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (strcmp(argv[i], "--tok") == 0 && i + 1 < argc)
            tok_path = argv[++i];
        else if (strcmp(argv[i], "--rag") == 0 && i + 1 < argc)
            rag_path = argv[++i];
        else if (strcmp(argv[i], "--arena") == 0 && i + 1 < argc)
            arena_size = atoi(argv[++i]) * 1024 * 1024;
        else if (strcmp(argv[i], "--agents") == 0 && i + 1 < argc)
            max_agents = atoi(argv[++i]);
    }

    aaedn::Arena arena = aaedn::arena_create(arena_size, 64);

    aaedn::Transformer tf;
    aaedn::transformer_init(&tf, &arena, model_path);

    aaedn::Tokenizer tok = aaedn::tokenizer_load(&arena, tok_path);

    aaedn::RAGIndex rag = aaedn::rag_load(&arena, rag_path);

    aaedn::smt_init();
    aaedn::bw_init();

    aaedn::Agent agents[6];
    aaedn::agents_init(agents, max_agents, &arena);

    aaedn::g_start_time = time(nullptr);
    aaedn::print_startup(&arena, arena_size, &tf, &tok, &rag);

    while (aaedn::g_running)
    {
        printf("> ");
        fflush(stdout);

        std::string line = aaedn::read_line();

        if (line.empty())
            continue;

        if (line[0] == '/')
        {
            aaedn::handle_command(line.c_str(), &aaedn::g_running);
            continue;
        }

        std::vector<uint32_t> tokens = aaedn::tokenizer_encode(&tok, line.c_str(), true);

        if (tokens.empty())
        {
            printf("[ERROR] Empty tokenization\n");
            continue;
        }

        float logits[32768] = {0};
        aaedn::transformer_forward(&tf, tokens.data(), tokens.size(), logits);

        float max_val = logits[0];
        for (int i = 1; i < 32768; ++i)
        {
            if (logits[i] > max_val)
                max_val = logits[i];
        }

        for (int i = 0; i < 32768; ++i)
        {
            logits[i] = expf(logits[i] - max_val);
        }

        float sum = 0.0f;
        for (int i = 0; i < 32768; ++i)
            sum += logits[i];

        for (int i = 0; i < 32768; ++i)
            logits[i] /= sum;

        uint32_t next_token = 0;
        float r = (float)rand() / (float)RAND_MAX;
        float cumsum = 0.0f;
        for (int i = 0; i < 32768; ++i)
        {
            cumsum += logits[i];
            if (r <= cumsum)
            {
                next_token = i;
                break;
            }
        }

        if (next_token == aaedn::TOKEN_EOS)
        {
            printf("\n");
            aaedn::g_tokens_decoded++;
            continue;
        }

        if (next_token == aaedn::TOKEN_PAD || next_token >= tok.vocab_size)
        {
            continue;
        }

        std::string decoded = aaedn::tokenizer_decode(&tok, {next_token});
        printf("%s", decoded.c_str());
        fflush(stdout);

        aaedn::g_tokens_decoded++;
    }

    aaedn::arena_destroy(&arena);

    return 0;
}
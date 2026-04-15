#include "../include/logger.hpp"
#include <csignal>
#include <cstdio>
#include <ctime>
#include <mutex>

namespace aaedn
{

static FILE* g_log = nullptr;
static LogLevel g_level = LogLevel::INFO;
static std::mutex g_mu;

static const char* level_name(LogLevel level)
{
    if (level == LogLevel::DEBUG)
        return "DEBUG";
    if (level == LogLevel::INFO)
        return "INFO";
    if (level == LogLevel::WARN)
        return "WARN";
    return "ERROR";
}

static void write_line(LogLevel level, const char* message)
{
    if (!g_log || !message)
        return;
    std::time_t now = std::time(nullptr);
    std::tm tmv{};
    localtime_r(&now, &tmv);
    std::fprintf(g_log, "%04d-%02d-%02dT%02d:%02d:%02dZ [%s] %s\n", tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
                 tmv.tm_hour, tmv.tm_min, tmv.tm_sec, level_name(level), message);
    std::fflush(g_log);
}

static void crash_handler(int sig)
{
    const char* message = "fatal signal captured";
    if (sig == SIGSEGV)
        message = "fatal signal SIGSEGV";
    else if (sig == SIGABRT)
        message = "fatal signal SIGABRT";
    else if (sig == SIGILL)
        message = "fatal signal SIGILL";
    else if (sig == SIGFPE)
        message = "fatal signal SIGFPE";
    write_line(LogLevel::ERROR, message);
    std::_Exit(128 + sig);
}

bool logger_init(const char* log_path, LogLevel level)
{
    std::lock_guard<std::mutex> lock(g_mu);
    if (!log_path)
        return false;
    FILE* f = std::fopen(log_path, "a");
    if (!f)
        return false;
    g_log = f;
    g_level = level;
    write_line(LogLevel::INFO, "logger initialized");
    return true;
}

void logger_set_level(LogLevel level)
{
    std::lock_guard<std::mutex> lock(g_mu);
    g_level = level;
}

void logger_log(LogLevel level, const char* message)
{
    std::lock_guard<std::mutex> lock(g_mu);
    if ((int)level < (int)g_level)
        return;
    write_line(level, message);
}

void logger_flush()
{
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_log)
        std::fflush(g_log);
}

void logger_install_crash_handler()
{
    std::signal(SIGSEGV, crash_handler);
    std::signal(SIGABRT, crash_handler);
    std::signal(SIGILL, crash_handler);
    std::signal(SIGFPE, crash_handler);
}

void logger_shutdown()
{
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_log)
        return;
    write_line(LogLevel::INFO, "logger shutdown");
    std::fclose(g_log);
    g_log = nullptr;
}

} // namespace aaedn

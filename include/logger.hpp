#ifndef AAEDN_LOGGER_HPP
#define AAEDN_LOGGER_HPP

#include <cstddef>

namespace aaedn
{

enum class LogLevel : int
{
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3
};

bool logger_init(const char* log_path, LogLevel level);
void logger_set_level(LogLevel level);
void logger_log(LogLevel level, const char* message);
void logger_flush();
void logger_install_crash_handler();
void logger_shutdown();

} // namespace aaedn

#endif

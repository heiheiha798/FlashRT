#ifndef FLASHRT_PROVIDERS_LLAMA_CPP_CHECKPOINT_IDENTITY_H
#define FLASHRT_PROVIDERS_LLAMA_CPP_CHECKPOINT_IDENTITY_H

#include <string>

namespace flashrt::providers::llama_cpp {

bool checkpoint_identity(const char* path, std::string* identity,
                         std::string* error);

}  // namespace flashrt::providers::llama_cpp

#endif

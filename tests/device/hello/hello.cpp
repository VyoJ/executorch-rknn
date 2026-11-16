// Toolchain smoke test: C++17 library features ExecuTorch relies on, threads,
// float math, and dynamic linking against the board's libstdc++ / uClibc.
#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>
#include <string_view>
#include <thread>
#include <variant>

int main() {
  std::optional<int> o = 42;
  std::variant<int, float> v = 1.5f;
  std::string_view sv = "string_view";
  auto p = std::make_unique<int>(7);
  int from_thread = 0;
  std::thread t([&] { from_thread = 1; });
  t.join();
  float e = std::exp(1.0f);

  std::printf("hello from luckfox: gcc %d.%d.%d, __cplusplus=%ld\n",
              __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__, (long)__cplusplus);
  std::printf("optional=%d variant=%.1f sv=%s unique=%d thread=%d expf(1)=%.5f\n",
              *o, std::get<float>(v), sv.data(), *p, from_thread, e);
#if defined(__ARM_NEON)
  std::printf("neon=yes\n");
#else
  std::printf("neon=no\n");
#endif
  return 0;
}

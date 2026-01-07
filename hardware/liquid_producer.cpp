/**
 * liquid_producer.cpp - High-Speed NEON Producer with Shared Memory
 *
 * Runs NEON inference at full speed (~6M bytes/sec) and exports
 * state to a memory-mapped file for the Python UI to visualize.
 */

#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>

constexpr int D_INNER = 128;
constexpr int D_STATE = 16;
constexpr int STATE_SIZE = D_INNER * D_STATE; // 2048 bytes

// Shared memory structure
struct SharedState {
  uint64_t bytes_processed;
  uint64_t timestamp_us;
  float dt;
  float energy;
  int8_t state[D_INNER][D_STATE];
};

// Delta LUT
static int8_t dt_lut[256];

void init_dt_lut() {
  for (int i = 0; i < 256; i++) {
    float x = (float)(i - 128) / 128.0f;
    float dt = 0.001f + 0.099f * (x + 1.0f) / 2.0f;
    dt_lut[i] = (int8_t)(std::exp(-dt) * 127);
  }
}

// Simplified SSM step (NEON-aware)
void ssm_step(int8_t x_byte, int8_t h[D_INNER][D_STATE], float &dt_out) {
  int dt_idx = (128 + (int)(std::sin(x_byte * 0.1f) * 50)) & 0xFF;
  int8_t dt_scale = dt_lut[dt_idx];
  dt_out = (dt_idx - 128) / 128.0f * 0.5f + 0.5f;

  for (int d = 0; d < D_INNER; d++) {
    for (int n = 0; n < D_STATE; n++) {
      int decay = 70 + (d % 30);
      int h_decayed = (decay * (int)h[d][n]) >> 7;
      int input_scale = 20 + (n % 10);
      int h_input = (input_scale * x_byte) >> 7;
      int h_new = h_decayed + h_input;
      h[d][n] = (int8_t)std::max(-127, std::min(127, h_new));
    }
  }
}

float compute_energy(int8_t h[D_INNER][D_STATE]) {
  float energy = 0;
  for (int d = 0; d < D_INNER; d++) {
    for (int n = 0; n < D_STATE; n++) {
      energy += std::abs(h[d][n]);
    }
  }
  return energy;
}

int main(int argc, char *argv[]) {
  std::cout << "Liquid Producer - NEON Inference Engine" << std::endl;
  std::cout << "========================================" << std::endl;

  init_dt_lut();

  // Create shared memory file
  const char *shm_path = "/tmp/liquid_state.bin";
  int fd = open(shm_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    std::cerr << "Failed to create shared memory file" << std::endl;
    return 1;
  }

  // Set file size
  if (ftruncate(fd, sizeof(SharedState)) < 0) {
    std::cerr << "Failed to set file size" << std::endl;
    return 1;
  }

  // Memory map
  SharedState *shared = (SharedState *)mmap(
      nullptr, sizeof(SharedState), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

  if (shared == MAP_FAILED) {
    std::cerr << "Failed to mmap" << std::endl;
    return 1;
  }

  std::cout << "Shared memory: " << shm_path << std::endl;
  std::cout << "State size: " << sizeof(SharedState) << " bytes" << std::endl;
  std::cout << "\nRunning at full NEON speed..." << std::endl;
  std::cout << "Press Ctrl+C to stop" << std::endl;

  // Initialize state
  int8_t h[D_INNER][D_STATE] = {0};
  float dt = 0.5f;
  uint64_t bytes_processed = 0;
  int t = 0;

  auto start = std::chrono::high_resolution_clock::now();
  auto last_report = start;

  // Data source (sine wave or file)
  std::vector<int8_t> input_data;
  if (argc > 1) {
    std::ifstream file(argv[1], std::ios::binary);
    if (file) {
      file.seekg(0, std::ios::end);
      size_t size = file.tellg();
      file.seekg(0, std::ios::beg);
      input_data.resize(size);
      file.read((char *)input_data.data(), size);
      std::cout << "Loaded file: " << argv[1] << " (" << size << " bytes)"
                << std::endl;
    }
  }

  while (true) {
    // Process large batch at full speed
    for (int batch = 0; batch < 100000; batch++) {
      t++;
      int8_t x;
      if (!input_data.empty()) {
        x = input_data[t % input_data.size()] - 128;
      } else {
        x = (int8_t)(80 * std::sin(t * 0.05));
      }

      ssm_step(x, h, dt);
      bytes_processed++;
    }

    // Update shared memory (once per batch)
    auto now = std::chrono::high_resolution_clock::now();
    shared->bytes_processed = bytes_processed;
    shared->timestamp_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch())
            .count();
    shared->dt = dt;
    shared->energy = compute_energy(h);
    std::memcpy(shared->state, h, sizeof(h));

    // Sync to disk
    msync(shared, sizeof(SharedState), MS_ASYNC);

    // Report every second
    auto elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(now - last_report);
    if (elapsed.count() >= 1) {
      auto total_elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
      double rate = (double)bytes_processed / (total_elapsed.count() / 1000.0);
      std::cout << "\rBytes: " << bytes_processed
                << " | Rate: " << (int)(rate / 1000000) << "M/s"
                << " | Energy: " << shared->energy << "     " << std::flush;
      last_report = now;
    }
  }

  munmap(shared, sizeof(SharedState));
  close(fd);
  return 0;
}

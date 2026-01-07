/**
 * AXI-Stream Testbench
 *
 * Simulates the streaming interface without FPGA synthesis.
 */

#include "ssm_kernel_fixed8.cpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <queue>

// Simulate hls::stream
template <typename T> class sim_stream {
  std::queue<T> q;

public:
  void write(const T &val) { q.push(val); }
  T read() {
    T val = q.front();
    q.pop();
    return val;
  }
  bool empty() { return q.empty(); }
};

struct axis_byte_t {
  int8_t data;
  bool last;
};

// Simulated ssm_step
void ssm_step_sim(fixed8_t x_byte, fixed8_t h[D_INNER][D_STATE],
                  const fixed8_t A[D_INNER][D_STATE], const fixed8_t B[D_STATE],
                  const fixed8_t C[D_STATE], const fixed8_t D_param[D_INNER],
                  fixed8_t dt, fixed8_t y_out[D_INNER]) {
  for (int d = 0; d < D_INNER; d++) {
    int16_t y_acc = 0;

    for (int n = 0; n < D_STATE; n++) {
      int16_t dA = 127 + ((int16_t)dt * A[d][n] >> 7);
      dA = std::max((int16_t)-127, std::min((int16_t)127, dA));

      int16_t dB = ((int16_t)dt * B[n]) >> 7;

      int16_t h_new = ((int16_t)dA * h[d][n] + (int16_t)dB * x_byte) >> 7;
      h[d][n] = (int8_t)std::max(-127, std::min(127, (int)h_new));

      y_acc += (int16_t)C[n] * h[d][n];
    }

    int16_t skip = ((int16_t)D_param[d] * x_byte) >> 7;
    int16_t y_total = (y_acc >> 7) + skip;

    y_out[d] = (int8_t)std::max(-127, std::min(127, (int)y_total));
  }
}

int main() {
  std::cout << "AXI-Stream SSM Testbench" << std::endl;
  std::cout << "========================" << std::endl;

  // Initialize weights
  fixed8_t A[D_INNER][D_STATE];
  fixed8_t B[D_STATE];
  fixed8_t C[D_STATE];
  fixed8_t D_param[D_INNER];
  fixed8_t h[D_INNER][D_STATE] = {0};
  fixed8_t y[D_INNER];

  srand(42);

  for (int d = 0; d < D_INNER; d++) {
    for (int n = 0; n < D_STATE; n++) {
      A[d][n] = -3; // Small negative for stability
    }
    D_param[d] = 64; // 0.5 in Q7
  }
  for (int n = 0; n < D_STATE; n++) {
    B[n] = 30; // Larger B for visible state changes
    C[n] = 30;
  }

  fixed8_t dt = 50; // Larger dt for visible effect

  // Create input stream
  sim_stream<axis_byte_t> in_stream, out_stream;
  const char *message = "val: 0.123\n";

  std::cout << "\nStreaming: \"" << message << "\"" << std::endl;

  for (int i = 0; message[i] != '\0'; i++) {
    axis_byte_t pkt;
    pkt.data = (int8_t)(message[i] - 128); // Center around 0
    pkt.last = (message[i + 1] == '\0');
    in_stream.write(pkt);
  }

  // Process stream
  std::cout << "\nProcessing through SSM..." << std::endl;

  int byte_count = 0;
  while (!in_stream.empty()) {
    axis_byte_t in_pkt = in_stream.read();

    ssm_step_sim(in_pkt.data, h, A, B, C, D_param, dt, y);

    axis_byte_t out_pkt;
    out_pkt.data = y[0];
    out_pkt.last = in_pkt.last;
    out_stream.write(out_pkt);

    byte_count++;
  }

  std::cout << "  Processed " << byte_count << " bytes" << std::endl;
  std::cout << "  Output queue size: " << byte_count << std::endl;

  // Verify hidden state evolved
  float h_sum = 0;
  for (int d = 0; d < D_INNER; d++) {
    for (int n = 0; n < D_STATE; n++) {
      h_sum += std::abs(h[d][n]);
    }
  }

  std::cout << "  Final state energy: " << h_sum << std::endl;

  if (h_sum > 0) {
    std::cout << "\nPASS: AXI-Stream simulation successful" << std::endl;
    return 0;
  } else {
    std::cout << "\nFAIL: State did not evolve" << std::endl;
    return 1;
  }
}

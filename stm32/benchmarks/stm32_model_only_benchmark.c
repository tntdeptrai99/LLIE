#include "stm32_model_only_benchmark.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "main.h"

/*
 * Adjust this block if STM32Cube.AI generated a different network name.
 * The default generated network is usually named "network".
 */
#include "network.h"
#include "network_data.h"

#define BENCHMARK_CPU_HZ 480000000.0f
#define BENCHMARK_WARMUP_RUNS 20U
#define BENCHMARK_MEASURED_RUNS 200U

#ifndef AI_NETWORK_IN_1_SIZE_BYTES
#define AI_NETWORK_IN_1_SIZE_BYTES (AI_NETWORK_IN_1_SIZE * sizeof(ai_i8))
#endif

#ifndef AI_NETWORK_OUT_1_SIZE_BYTES
#define AI_NETWORK_OUT_1_SIZE_BYTES (AI_NETWORK_OUT_1_SIZE * sizeof(ai_i8))
#endif

static ai_handle network = AI_HANDLE_NULL;
static AI_ALIGNED(32) ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];
static AI_ALIGNED(32) ai_i8 input_data[AI_NETWORK_IN_1_SIZE];
static AI_ALIGNED(32) ai_i8 output_data[AI_NETWORK_OUT_1_SIZE];
static float measured_ms[BENCHMARK_MEASURED_RUNS];

static void dwt_counter_init(void)
{
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static uint32_t dwt_cycles(void)
{
  return DWT->CYCCNT;
}

static float cycles_to_ms(uint32_t cycles)
{
  return ((float)cycles * 1000.0f) / BENCHMARK_CPU_HZ;
}

static int float_cmp(const void *a, const void *b)
{
  const float fa = *(const float *)a;
  const float fb = *(const float *)b;
  return (fa > fb) - (fa < fb);
}

static void fill_representative_input(void)
{
  /*
   * Deterministic dark-ish tensor. Replace with a captured/calibration input
   * later if you want scene-specific latency checks; keep preprocessing outside
   * this benchmark.
   */
  for (uint32_t i = 0; i < AI_NETWORK_IN_1_SIZE; ++i) {
    input_data[i] = (ai_i8)((i * 13U + 17U) % 64U - 32);
  }
}

static int network_init_once(void)
{
  ai_error err;
  const ai_handle acts[] = { activations };
  ai_network_params params = {
    AI_NETWORK_DATA_WEIGHTS(ai_network_data_weights_get()),
    AI_NETWORK_DATA_ACTIVATIONS(acts)
  };

  err = ai_network_create(&network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    printf("ai_network_create failed: type=%d code=%d\r\n", err.type, err.code);
    return 0;
  }

  if (!ai_network_init(network, &params)) {
    err = ai_network_get_error(network);
    printf("ai_network_init failed: type=%d code=%d\r\n", err.type, err.code);
    return 0;
  }

  return 1;
}

static int network_run_once(void)
{
  ai_i32 nbatch;
  ai_buffer ai_input[AI_NETWORK_IN_NUM] = AI_NETWORK_IN;
  ai_buffer ai_output[AI_NETWORK_OUT_NUM] = AI_NETWORK_OUT;

  ai_input[0].data = AI_HANDLE_PTR(input_data);
  ai_output[0].data = AI_HANDLE_PTR(output_data);

  nbatch = ai_network_run(network, ai_input, ai_output);
  if (nbatch != 1) {
    ai_error err = ai_network_get_error(network);
    printf("ai_network_run failed: nbatch=%ld type=%d code=%d\r\n",
           (long)nbatch, err.type, err.code);
    return 0;
  }

  return 1;
}

static void summarize_and_print(void)
{
  float sorted[BENCHMARK_MEASURED_RUNS];
  float sum = 0.0f;
  float min_v = measured_ms[0];
  float max_v = measured_ms[0];

  memcpy(sorted, measured_ms, sizeof(sorted));
  qsort(sorted, BENCHMARK_MEASURED_RUNS, sizeof(float), float_cmp);

  for (uint32_t i = 0; i < BENCHMARK_MEASURED_RUNS; ++i) {
    const float v = measured_ms[i];
    sum += v;
    if (v < min_v) {
      min_v = v;
    }
    if (v > max_v) {
      max_v = v;
    }
  }

  const float mean = sum / (float)BENCHMARK_MEASURED_RUNS;
  float variance = 0.0f;
  for (uint32_t i = 0; i < BENCHMARK_MEASURED_RUNS; ++i) {
    const float d = measured_ms[i] - mean;
    variance += d * d;
  }
  variance /= (float)BENCHMARK_MEASURED_RUNS;

  const float median = sorted[BENCHMARK_MEASURED_RUNS / 2U];
  const uint32_t p95_idx = (uint32_t)((BENCHMARK_MEASURED_RUNS * 95U + 99U) / 100U) - 1U;
  const float p95 = sorted[p95_idx];
  const float std = sqrtf(variance);
  const float fps_mean = 1000.0f / mean;

  printf("\r\nMODEL_ONLY_INT8_QDQ_TRIAL011_96\r\n");
  printf("warmup_runs,%lu\r\n", (unsigned long)BENCHMARK_WARMUP_RUNS);
  printf("measured_runs,%lu\r\n", (unsigned long)BENCHMARK_MEASURED_RUNS);
  printf("cpu_hz,%.0f\r\n", BENCHMARK_CPU_HZ);
  printf("input_size_bytes,%lu\r\n", (unsigned long)AI_NETWORK_IN_1_SIZE_BYTES);
  printf("output_size_bytes,%lu\r\n", (unsigned long)AI_NETWORK_OUT_1_SIZE_BYTES);
  printf("activation_size_bytes,%lu\r\n", (unsigned long)AI_NETWORK_DATA_ACTIVATIONS_SIZE);
  printf("mean_ms,%.3f\r\n", mean);
  printf("median_ms,%.3f\r\n", median);
  printf("p95_ms,%.3f\r\n", p95);
  printf("min_ms,%.3f\r\n", min_v);
  printf("max_ms,%.3f\r\n", max_v);
  printf("std_ms,%.3f\r\n", std);
  printf("fps_mean,%.3f\r\n", fps_mean);
  printf("pass_latency,%s\r\n", mean < 277.0f ? "yes" : "no");
  printf("pass_fps,%s\r\n", fps_mean > 3.6f ? "yes" : "no");
}

void stm32_model_only_benchmark_run(void)
{
  fill_representative_input();
  dwt_counter_init();

  if (!network_init_once()) {
    return;
  }

  printf("Starting model-only benchmark: INT8 QDQ trial011 96x96\r\n");

  for (uint32_t i = 0; i < BENCHMARK_WARMUP_RUNS; ++i) {
    if (!network_run_once()) {
      return;
    }
  }

  for (uint32_t i = 0; i < BENCHMARK_MEASURED_RUNS; ++i) {
    const uint32_t start = dwt_cycles();
    if (!network_run_once()) {
      return;
    }
    const uint32_t end = dwt_cycles();
    measured_ms[i] = cycles_to_ms(end - start);
  }

  summarize_and_print();
}

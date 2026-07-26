/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "crc.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "llieai.h"
#include "llieai_data.h"
#include "llieai_data_params.h"
#include "camera.h"
#include "dcmi.h"
#include "i2c.h"
#include "lcd.h"
#include "spi.h"
#include "tim.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef struct
{
  uint32_t camera_start_cycles;
  uint32_t camera_wait_cycles;
  uint32_t camera_dma_cycles;
  uint32_t camera_cache_cycles;
  uint32_t preprocess_cycles;
  uint32_t input_copy_cycles;
  uint32_t inference_cycles;
  uint32_t postprocess_cycles;
  uint32_t rgb565_cycles;
  uint32_t lcd_prepare_cycles;
  uint32_t lcd_cache_cycles;
  uint32_t lcd_transfer_cycles;
  uint32_t total_cycles;
} PipelineProfile;

typedef struct
{
  uint32_t min;
  uint32_t mean;
  uint32_t max;
  uint32_t p95;
} PipelineProfileStats;

typedef struct
{
  uint32_t min_value;
  uint32_t max_value;
  uint32_t checksum;
} TensorStats;

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
#define FRAME_WIDTH 160U
#define FRAME_HEIGHT 120U
#define MODEL_WIDTH 96U
#define MODEL_HEIGHT 96U
#define MODEL_CHANNELS 3U
#define MODEL_PLANE_SIZE (MODEL_WIDTH * MODEL_HEIGHT)
#define MODEL_TENSOR_SIZE (MODEL_PLANE_SIZE * MODEL_CHANNELS)
#define MODEL_DISPLAY_WIDTH 96U
#define MODEL_DISPLAY_HEIGHT 80U
#define DISPLAY_MODE_LCD_TEST 0U
#define DISPLAY_MODE_CAMERA_RAW 1U
#define DISPLAY_MODE_MODEL_INPUT 2U
#define DISPLAY_MODE_MODEL_OUTPUT 3U
#define DISPLAY_MODE_CONSTANT_INPUT_OUTPUT 4U
#define DISPLAY_MODE_CANDIDATE_OUTPUT 5U
#define DISPLAY_MODE_EXTERNAL_IO_OUTPUT 6U
#define DISPLAY_MODE_GAIN_MAP 7U
#define DISPLAY_MODE_RESIDUAL_MAP 8U
#define ENABLE_E2E_DEBUG_STATS 1U
#define ENABLE_AI_OUTPUT_DEBUG 1U
#define PIPELINE_CORRECTNESS_MODE 0U
#define TEST_LCD_PATTERN 0U
#define TEST_LCD_RGBRECT_STREAM_ONLY 0U
#define TEST_FAKE_AI_OUTPUT_TO_LCD 0U
#define TEST_FIXED_INPUT_AI 0U
#define TEST_FIXED_INPUT_AI_TO_LCD 0U
#define TEST_CAMERA_TO_AI_NO_LCD 0U
#define TEST_KEY_FINDER_ONLY 0U
#define TEST_AI_OUTPUT_LAYOUT_SWEEP 0U
#define TEST_FULL_PIPELINE 1U
#define TEST_FULL_PIPELINE_ONE_SHOT 0U
#define TEST_BLOCKING_SNAPSHOT_LOOP 1U
#define TEST_CUBEAI_FIXED_INPUT_ONLY 0U
#define TEST_CAMERA_ID_ONLY 0U
#define TEST_LCD_AFTER_XCLK_ONLY 0U
#define TEST_LCD_EARLY_ONLY 0U
#define ENABLE_PIPELINE_PROFILING 0U
#define ENABLE_TENSOR_EQUIV_DUMP 0U
#define ENABLE_LCD_RGB565_DUMP 0U
#define ENABLE_EXTERNAL_IO_MODE 0U
#define ENABLE_LEGACY_DEBUG_DISPLAY_MODES 0U
#define CAMERA_RGB565_BYTE_SWAP_FOR_AI 1U
#define CAMERA_RGB565_RB_SWAP_FOR_AI 0U
#define LCD_RGB565_STORE_BIG_ENDIAN 1U
#define CAMERA_STOP_DURING_PROCESSING 1U
#define DISPLAY_SAFE_INPUT_AFTER_AI 0U
#define PIPELINE_PROFILE_FRAME_COUNT 100U
#define FW_PROBE_VERSION 7305U
#define LLIEAI_INPUT_ACTIVATION_OFFSET 314664U
#define LLIEAI_OUTPUT_OFFSET 27648U
#define LLIEAI_OUTPUT_CANDIDATE_OFFSET 55296U
#define LLIEAI_OUTPUT_CANDIDATE2_OFFSET 27648U
#define LLIEAI_OUTPUT_CANDIDATE3_OFFSET 55376U
#define LLIEAI_OUTPUT_CANDIDATE4_OFFSET 165888U
#define LLIEAI_OUTPUT0_ACTIVATION_OFFSET 0U
#define LLIEAI_OUTPUT1_ACTIVATION_OFFSET 138240U
#define LLIEAI_PRETRANSPOSE_OUTPUT_OFFSET 110592U
#define LLIEAI_GAIN_ACTIVATION_OFFSET LLIEAI_OUTPUT0_ACTIVATION_OFFSET
#define LLIEAI_RESIDUAL_ACTIVATION_OFFSET LLIEAI_OUTPUT1_ACTIVATION_OFFSET
#define MODEL_IO_HWC_INDEX(x, y, c) (((((y) * MODEL_WIDTH) + (x)) * MODEL_CHANNELS) + (c))
#define MODEL_INPUT_HWC_INDEX(x, y, c) MODEL_IO_HWC_INDEX(x, y, c)
#define MODEL_IO_CHW_YX_INDEX(x, y, c) (((c) * MODEL_PLANE_SIZE) + ((y) * MODEL_WIDTH) + (x))
#define MODEL_IO_CHW_XY_INDEX(x, y, c) (((c) * MODEL_PLANE_SIZE) + ((x) * MODEL_HEIGHT) + (y))
#define MODEL_IO_PUBLIC_YCX_INDEX(x, y, c) (((((y) * MODEL_CHANNELS) + (c)) * MODEL_WIDTH) + (x))
#define MODEL_COMPOSED_OUTPUT_OFFSET 0x5441494CU
AI_ALIGNED(32) static ai_u8 ai_activations_d1[AI_LLIEAI_DATA_ACTIVATION_1_SIZE] __attribute__((section(".ai_activations_d1"), aligned(32)));
#if AI_LLIEAI_DATA_ACTIVATIONS_COUNT >= 2
AI_ALIGNED(32) static ai_u8 ai_activations_d2[AI_LLIEAI_DATA_ACTIVATION_2_SIZE] __attribute__((section(".ram_d2"), aligned(32)));
#endif
#if TEST_CUBEAI_FIXED_INPUT_ONLY || !defined(AI_LLIEAI_INPUTS_IN_ACTIVATIONS) || (AI_LLIEAI_INPUTS_IN_ACTIVATIONS == 0)
AI_ALIGNED(4) static ai_u8 ai_input_data[AI_LLIEAI_IN_1_SIZE_BYTES];
#endif
AI_ALIGNED(4) static ai_u8 ai_output_data[AI_LLIEAI_OUT_1_SIZE_BYTES];
#if (AI_LLIEAI_OUT_NUM >= 2) && (!defined(AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS) || (AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS == 0))
AI_ALIGNED(4) static ai_u8 ai_tail_gain_data[MODEL_TENSOR_SIZE] __attribute__((section(".ram_d2"), aligned(32)));
AI_ALIGNED(4) static ai_u8 ai_tail_residual_data[MODEL_TENSOR_SIZE] __attribute__((section(".ram_d2"), aligned(32)));
#endif

static ai_handle ai_network = AI_HANDLE_NULL;
static ai_buffer *ai_input;
static ai_buffer *ai_output;
static ai_buffer ai_run_input[AI_LLIEAI_IN_NUM];
static ai_buffer ai_run_output[AI_LLIEAI_OUT_NUM];
static ai_u8 *ai_runtime_input;
static ai_u8 *ai_runtime_output;
static ai_u8 *ai_network_output;
static ai_u8 *ai_runtime_gain;
static ai_u8 *ai_runtime_residual;

static uint16_t camera_buf_0[FRAME_WIDTH][FRAME_HEIGHT] __attribute__((section(".ram_d2"), aligned(32)));
static uint16_t camera_buf_1[FRAME_WIDTH][FRAME_HEIGHT] __attribute__((section(".ram_d2"), aligned(32)));
static uint16_t lcd_buf_0[MODEL_DISPLAY_WIDTH * MODEL_DISPLAY_HEIGHT] __attribute__((aligned(32)));
static uint16_t lcd_buf_1[MODEL_DISPLAY_WIDTH * MODEL_DISPLAY_HEIGHT] __attribute__((aligned(32)));

volatile uint16_t* camera_dma_buf = (volatile uint16_t*)camera_buf_0;
volatile uint16_t* camera_proc_buf = (volatile uint16_t*)camera_buf_1;
volatile uint16_t* lcd_front_buf = (volatile uint16_t*)lcd_buf_0;
volatile uint16_t* lcd_back_buf = (volatile uint16_t*)lcd_buf_1;
static ai_u8 model_input_shadow[MODEL_TENSOR_SIZE] __attribute__((section(".ram_d2"), aligned(32)));
static ai_u8 model_input_runtime_snapshot[MODEL_TENSOR_SIZE] __attribute__((section(".ram_d2"), aligned(32)));

static const ai_u8 residual_tail_lut[256] = {
  0,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,
  31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,
  63,65,67,69,71,73,75,75,77,79,81,83,85,87,89,91,
  93,95,97,99,101,103,105,107,109,111,113,115,117,119,121,123,
  125,127,129,131,133,135,137,139,141,143,145,147,149,151,151,153,
  155,157,159,161,163,165,167,169,171,173,175,177,179,181,183,185,
  187,189,191,193,195,197,199,201,203,205,207,209,211,213,215,217,
  219,221,223,225,227,227,229,231,233,235,237,239,241,243,245,247,
  249,251,253,255,255,255,255,255,255,255,255,255,255,255,255,255,
  255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
  255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
  255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
  255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
  255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
  255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
  255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
};

volatile uint32_t camera_display_ms = 0;
volatile uint32_t preprocess_ms = 0;
volatile uint32_t inference_ms = 0;
volatile uint32_t postprocess_ms = 0;
volatile uint32_t total_ms = 0;
volatile uint32_t total_fps = 0;
volatile uint32_t E2E_FPS = 0;
volatile uint32_t E2E_AvgMs = 0;
volatile uint32_t E2E_LastMs = 0;
volatile uint32_t boot_stage = 0;
volatile uint32_t error_stage = 0;
volatile uint32_t clock_config_source = 0;
volatile uint32_t camera_detected_id = 0;
volatile uint32_t camera_tim_detected_id = 0;
volatile uint32_t camera_mco_detected_id = 0;
volatile uint32_t display_mode = DISPLAY_MODE_LCD_TEST;
volatile uint32_t ai_input_layout = 1;
volatile uint32_t ai_output_layout = 1;
volatile uint32_t ai_output_layout_sweep_frame = 0;
volatile uint32_t display_ab_mode = 0;
static uint32_t key_last_ms = 0;
static GPIO_PinState key_last_pa2_state = GPIO_PIN_SET;
static GPIO_PinState key_last_pc13_state = GPIO_PIN_SET;
static GPIO_PinState key_last_pe3_state = GPIO_PIN_SET;
static volatile uint32_t key_irq_pending = 0U;
static volatile uint32_t key_irq_pin = 0U;
#define KEY_PROBE_COUNT 13U
static GPIO_PinState key_probe_last[KEY_PROBE_COUNT];
static uint16_t key_probe_last_pa = 0U;
static uint16_t key_probe_last_pb = 0U;
static uint16_t key_probe_last_pc = 0U;
static uint16_t key_probe_last_pd = 0U;
static uint16_t key_probe_last_pe = 0U;
volatile uint32_t camera_raw_min = 0;
volatile uint32_t camera_raw_max = 0;
volatile uint32_t model_input_min = 0;
volatile uint32_t model_input_max = 0;
volatile uint32_t model_output_min = 0;
volatile uint32_t model_output_max = 0;
volatile uint32_t postprocess_gain_q8 = 256;
volatile uint32_t postprocess_residual_q8 = 256;
volatile uint32_t display_fallback_count = 0;
volatile uint32_t ai_input_addr = 0;
volatile uint32_t ai_output_addr = 0;
volatile uint32_t active_output_offset = 0;
volatile uint32_t out0_min = 0;
volatile uint32_t out0_max = 0;
volatile uint32_t out55296_min = 0;
volatile uint32_t out55296_max = 0;
volatile uint32_t out27648_min = 0;
volatile uint32_t out27648_max = 0;
volatile uint32_t out55376_min = 0;
volatile uint32_t out55376_max = 0;
volatile uint32_t out165888_min = 0;
volatile uint32_t out165888_max = 0;
volatile uint32_t external_output_min = 0;
volatile uint32_t external_output_max = 0;
volatile uint32_t external_input_min = 0;
volatile uint32_t external_input_max = 0;
volatile uint32_t fw_probe_version = FW_PROBE_VERSION;
volatile uint32_t ai_input_runtime_offset = 0xFFFFFFFFU;
volatile uint32_t ai_output_runtime_offset = 0xFFFFFFFFU;
volatile uint32_t model_output1_min = 0;
volatile uint32_t model_output1_max = 0;
volatile uint32_t act110_min = 0;
volatile uint32_t act110_max = 0;
volatile uint32_t act222_min = 0;
volatile uint32_t act222_max = 0;
volatile uint32_t act279_min = 0;
volatile uint32_t act279_max = 0;
volatile uint32_t camera_dma_enabled = 0;
volatile uint32_t tensor_dump_request = 0;
volatile uint32_t tensor_dump_done = 0;
volatile uint32_t tensor_dump_frames = 0;

static volatile uint8_t dcmi_frame_ready = 0;
static volatile uint32_t dcmi_error_count = 0;

#if ENABLE_AI_OUTPUT_DEBUG
static TensorStats debug_out0_stats;
static TensorStats debug_out1_stats;
static uint32_t debug_lcd_post_checksum = 0;
static uint32_t debug_fixed_runs = 0;
static uint32_t debug_out0_ref = 0;
static uint32_t debug_out1_ref = 0;
static uint32_t debug_lcd_ref = 0;
static uint32_t debug_out0_mismatch = 0;
static uint32_t debug_out1_mismatch = 0;
static uint32_t debug_lcd_mismatch = 0;
#endif

#if ENABLE_PIPELINE_PROFILING
static PipelineProfile pipeline_profiles[PIPELINE_PROFILE_FRAME_COUNT];
PipelineProfile pipeline_profile_last;
volatile uint32_t profile_frames_collected = 0;
volatile uint32_t profile_report_ready = 0;
static volatile uint32_t profile_camera_start_overhead_cycles = 0;
static volatile uint32_t profile_camera_dma_cycles = 0;
static volatile uint32_t profile_camera_dma_valid = 0;
static volatile uint32_t profile_camera_dma_mark_cycles = 0;
static uint32_t profile_wait_start_cycles = 0;
#endif

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MPU_Config(void);
/* USER CODE BEGIN PFP */
static void MX_DMA_Init(void);
static uint32_t model_io_index(uint32_t x, uint32_t y, uint32_t c, uint32_t layout);
static void preprocess_camera_to_ai_input_variant(uint32_t byte_swap, uint32_t rb_swap);
static uint16_t rgb888_to_rgb565(uint8_t r, uint8_t g, uint8_t b);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
int _write(int file, char *ptr, int len)
{
  (void)file;
  HAL_UART_Transmit(&huart1, (uint8_t *)ptr, (uint16_t)len, HAL_MAX_DELAY);
  return len;
}

int _read(int file, char *ptr, int len)
{
  (void)file;
  (void)ptr;
  (void)len;
  errno = ENOSYS;
  return -1;
}

int _close(int file)
{
  (void)file;
  errno = ENOSYS;
  return -1;
}

int _fstat(int file, struct stat *st)
{
  (void)file;
  st->st_mode = S_IFCHR;
  return 0;
}

int _isatty(int file)
{
  (void)file;
  return 1;
}

int _lseek(int file, int ptr, int dir)
{
  (void)file;
  (void)ptr;
  (void)dir;
  return 0;
}

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

static uint32_t activation_offset_of(const ai_u8 *ptr)
{
  const ai_u8 *base = ai_activations_d1;
  const ai_u8 *end = &ai_activations_d1[AI_LLIEAI_DATA_ACTIVATION_1_SIZE];

  if ((ptr >= base) && (ptr < end)) {
    return (uint32_t)(ptr - base);
  }
#if AI_LLIEAI_DATA_ACTIVATIONS_COUNT >= 2
  base = ai_activations_d2;
  end = &ai_activations_d2[AI_LLIEAI_DATA_ACTIVATION_2_SIZE];
  if ((ptr >= base) && (ptr < end)) {
    return 0xD2000000U | (uint32_t)(ptr - base);
  }
#endif
  return 0xFFFFFFFFU;
}

#if ENABLE_TENSOR_EQUIV_DUMP
static uint32_t fnv1a_u8(const ai_u8 *buffer, uint32_t length)
{
  uint32_t hash = 2166136261UL;

  for (uint32_t i = 0; i < length; ++i) {
    hash ^= buffer[i];
    hash *= 16777619UL;
  }

  return hash;
}

static void print_hex_byte(ai_u8 value)
{
  static const char hex[] = "0123456789ABCDEF";
  char out[2];

  out[0] = hex[(value >> 4) & 0x0FU];
  out[1] = hex[value & 0x0FU];
  HAL_UART_Transmit(&huart1, (uint8_t *)out, sizeof(out), HAL_MAX_DELAY);
}

static void dump_tensor_u8(const char *name, const ai_u8 *buffer, uint32_t length)
{
  const uint32_t chunk = 64U;
  static const char crlf[] = "\r\n";
  uint32_t checksum = fnv1a_u8(buffer, length);

  printf("tdump_tensor,%s,%lu,%08lX\r\n", name, length, checksum);

  for (uint32_t offset = 0; offset < length; offset += chunk) {
    uint32_t n = length - offset;
    if (n > chunk) {
      n = chunk;
    }

    printf("tdump_data,%s,%lu,", name, offset);
    fflush(stdout);
    for (uint32_t i = 0; i < n; ++i) {
      print_hex_byte(buffer[offset + i]);
    }
    HAL_UART_Transmit(&huart1, (uint8_t *)crlf, 2U, HAL_MAX_DELAY);
  }

  printf("tdump_end,%s\r\n", name);
}

static void dump_tensor_equivalence_frame(void)
{
  ++tensor_dump_frames;
  printf("tdump_begin,v=%lu,frame=%lu,in_layout=%lu,out_layout=%lu,in_len=%lu,out0_len=%lu,out1_len=%lu,in_off=%lu,out0_off=%lu,out1_off=%lu\r\n",
      fw_probe_version,
      tensor_dump_frames,
      ai_input_layout,
      ai_output_layout,
      (uint32_t)AI_LLIEAI_IN_1_SIZE_BYTES,
      (uint32_t)AI_LLIEAI_OUT_1_SIZE_BYTES,
#if AI_LLIEAI_OUT_NUM >= 2
      (uint32_t)AI_LLIEAI_OUT_2_SIZE_BYTES,
#else
      0UL,
#endif
      activation_offset_of(ai_runtime_input),
      activation_offset_of(ai_runtime_output),
#if AI_LLIEAI_OUT_NUM >= 2
      activation_offset_of(ai_runtime_residual));
#else
      0xFFFFFFFFUL);
#endif

  dump_tensor_u8("input_runtime", model_input_runtime_snapshot, AI_LLIEAI_IN_1_SIZE_BYTES);
  dump_tensor_u8("output0_public", ai_runtime_output, AI_LLIEAI_OUT_1_SIZE_BYTES);
#if AI_LLIEAI_OUT_NUM >= 2
  dump_tensor_u8("output1_public", ai_runtime_residual, AI_LLIEAI_OUT_2_SIZE_BYTES);
#endif
  printf("tdump_done,frame=%lu\r\n", tensor_dump_frames);
}

static void dump_input_decode_variants(void)
{
  preprocess_camera_to_ai_input_variant(0U, 0U);
  dump_tensor_u8("input_b0_rb0", ai_runtime_input, AI_LLIEAI_IN_1_SIZE_BYTES);
  preprocess_camera_to_ai_input_variant(1U, 0U);
  dump_tensor_u8("input_b1_rb0", ai_runtime_input, AI_LLIEAI_IN_1_SIZE_BYTES);
  preprocess_camera_to_ai_input_variant(0U, 1U);
  dump_tensor_u8("input_b0_rb1", ai_runtime_input, AI_LLIEAI_IN_1_SIZE_BYTES);
  preprocess_camera_to_ai_input_variant(1U, 1U);
  dump_tensor_u8("input_b1_rb1", ai_runtime_input, AI_LLIEAI_IN_1_SIZE_BYTES);
}
#endif

#if ENABLE_PIPELINE_PROFILING
static uint32_t profile_get_field(const PipelineProfile *profile, uint32_t field)
{
  switch (field) {
    case 0: return profile->camera_start_cycles;
    case 1: return profile->camera_wait_cycles;
    case 2: return profile->camera_dma_cycles;
    case 3: return profile->camera_cache_cycles;
    case 4: return profile->preprocess_cycles;
    case 5: return profile->input_copy_cycles;
    case 6: return profile->inference_cycles;
    case 7: return profile->postprocess_cycles;
    case 8: return profile->rgb565_cycles;
    case 9: return profile->lcd_prepare_cycles;
    case 10: return profile->lcd_cache_cycles;
    case 11: return profile->lcd_transfer_cycles;
    default: return profile->total_cycles;
  }
}

static PipelineProfileStats profile_calc_stats(uint32_t field)
{
  PipelineProfileStats stats = {0};
  uint32_t sorted[PIPELINE_PROFILE_FRAME_COUNT];
  uint64_t sum = 0;
  uint32_t count = profile_frames_collected;

  if (count > PIPELINE_PROFILE_FRAME_COUNT) {
    count = PIPELINE_PROFILE_FRAME_COUNT;
  }
  if (count == 0U) {
    return stats;
  }

  for (uint32_t i = 0; i < count; ++i) {
    uint32_t value = profile_get_field(&pipeline_profiles[i], field);
    sorted[i] = value;
    sum += value;
  }

  for (uint32_t i = 1; i < count; ++i) {
    uint32_t value = sorted[i];
    uint32_t j = i;
    while ((j > 0U) && (sorted[j - 1U] > value)) {
      sorted[j] = sorted[j - 1U];
      --j;
    }
    sorted[j] = value;
  }

  stats.min = sorted[0];
  stats.mean = (uint32_t)(sum / count);
  stats.max = sorted[count - 1U];
  stats.p95 = sorted[((count * 95U) + 99U) / 100U - 1U];
  return stats;
}

static void profile_print_one(uint32_t field)
{
  PipelineProfileStats stats = profile_calc_stats(field);
  uint32_t cycles_per_us = SystemCoreClock / 1000000U;

  if (cycles_per_us == 0U) {
    cycles_per_us = 1U;
  }

  printf("p,%lu,%lu,%lu,%lu,%lu\r\n",
      field,
      stats.min / cycles_per_us,
      stats.mean / cycles_per_us,
      stats.max / cycles_per_us,
      stats.p95 / cycles_per_us);
}

static void profile_store_frame(const PipelineProfile *profile)
{
  if (profile_report_ready != 0U) {
    return;
  }

  if (profile_frames_collected < PIPELINE_PROFILE_FRAME_COUNT) {
    pipeline_profiles[profile_frames_collected] = *profile;
    pipeline_profile_last = *profile;
    ++profile_frames_collected;
  }

  if (profile_frames_collected == PIPELINE_PROFILE_FRAME_COUNT) {
    profile_report_ready = 1U;
  }
}

static void profile_print_report_once(void)
{
  if (profile_report_ready != 1U) {
    return;
  }

  profile_report_ready = 2U;
  printf("\r\nprof,n=%lu,sys=%lu,h=%lu,p1=%lu,p2=%lu,clk=%lu\r\n",
      profile_frames_collected,
      SystemCoreClock,
      HAL_RCC_GetHCLKFreq(),
      HAL_RCC_GetPCLK1Freq(),
      HAL_RCC_GetPCLK2Freq(),
      clock_config_source);
  for (uint32_t field = 0; field <= 12U; ++field) {
    profile_print_one(field);
  }
}
#endif

static int llie_ai_init(void)
{
#if AI_LLIEAI_DATA_ACTIVATIONS_COUNT >= 2
  const ai_handle activations[] = {
    AI_HANDLE_PTR(ai_activations_d1),
    AI_HANDLE_PTR(ai_activations_d2)
  };
#else
  const ai_handle activations[] = {
    AI_HANDLE_PTR(ai_activations_d1)
  };
#endif

  ai_error err = ai_llieai_create_and_init(
      &ai_network,
      activations,
      AI_LLIEAI_DATA_WEIGHTS_TABLE_GET());

  if (err.type != AI_ERROR_NONE) {
    printf("ai init err %d/%d\r\n", err.type, err.code);
    return -1;
  }

  ai_input = ai_llieai_inputs_get(ai_network, NULL);
  ai_output = ai_llieai_outputs_get(ai_network, NULL);
  memcpy(ai_run_input, ai_input, sizeof(ai_run_input));
  memcpy(ai_run_output, ai_output, sizeof(ai_run_output));
#if defined(AI_LLIEAI_INPUTS_IN_ACTIVATIONS) && (AI_LLIEAI_INPUTS_IN_ACTIVATIONS != 0)
  ai_runtime_input = AI_BUFFER_DATA(&ai_input[0], ai_u8);
#else
  ai_runtime_input = ai_input_data;
#endif
#if defined(AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS) && (AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS != 0)
  ai_network_output = AI_BUFFER_DATA(&ai_output[0], ai_u8);
  ai_runtime_output = ai_output_data;
#if AI_LLIEAI_OUT_NUM >= 2
  ai_runtime_gain = ai_network_output;
  ai_runtime_residual = AI_BUFFER_DATA(&ai_output[1], ai_u8);
#endif
#if AI_LLIEAI_OUT_NUM < 2
  ai_runtime_gain = ai_runtime_output;
  ai_runtime_residual = ai_runtime_output;
#endif
#elif AI_LLIEAI_OUT_NUM >= 2
  ai_network_output = ai_output_data;
  ai_runtime_output = ai_output_data;
  ai_runtime_gain = ai_tail_gain_data;
  ai_runtime_residual = ai_tail_residual_data;
#else
  ai_network_output = ai_output_data;
  ai_runtime_output = ai_output_data;
  ai_runtime_gain = ai_runtime_output;
  ai_runtime_residual = ai_runtime_output;
#endif
  ai_run_input[0].data = AI_HANDLE_PTR(ai_runtime_input);
#if defined(AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS) && (AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS != 0)
  ai_run_output[0].data = AI_HANDLE_PTR(ai_network_output);
#if AI_LLIEAI_OUT_NUM >= 2
  ai_run_output[1].data = AI_HANDLE_PTR(ai_runtime_residual);
#endif
#elif AI_LLIEAI_OUT_NUM >= 2
  ai_run_output[0].data = AI_HANDLE_PTR(ai_runtime_gain);
  ai_run_output[1].data = AI_HANDLE_PTR(ai_runtime_residual);
#else
  ai_run_output[0].data = AI_HANDLE_PTR(ai_runtime_output);
#endif
  ai_input_addr = (uint32_t)ai_runtime_input;
  ai_output_addr = (uint32_t)ai_runtime_output;
  ai_input_runtime_offset = activation_offset_of(ai_runtime_input);
  ai_output_runtime_offset = activation_offset_of(ai_runtime_output);

  if ((ai_runtime_input == NULL) || (ai_runtime_output == NULL) ||
      (ai_runtime_gain == NULL) || (ai_runtime_residual == NULL)) {
    printf("ai io err\r\n");
    return -1;
  }

  return 0;
}

static void print_model_identity(void)
{
  printf("fw=%lu,m=d9fae00b\r\n", fw_probe_version);
  printf("aio=%lu/%lu/%lu,act=%lu\r\n",
      activation_offset_of(ai_runtime_input),
      activation_offset_of(ai_runtime_output),
#if AI_LLIEAI_OUT_NUM >= 2
      activation_offset_of(ai_runtime_residual),
#else
      0xFFFFFFFFUL,
#endif
      (uint32_t)AI_LLIEAI_DATA_ACTIVATIONS_SIZE);
}

static void fill_test_input(void)
{
  for (uint32_t i = 0; i < AI_LLIEAI_IN_1_SIZE_BYTES; ++i) {
    ai_runtime_input[i] = 32U;
    model_input_shadow[i] = 32U;
  }
}

static void __attribute__((unused)) run_llie_benchmark(void)
{
  const uint32_t warmup_runs = 10;
  const uint32_t measure_runs = 100;
  uint64_t sum_cycles = 0;
  uint32_t min_cycles = 0xFFFFFFFFU;
  uint32_t max_cycles = 0;

  fill_test_input();
  memset(ai_output_data, 0, sizeof(ai_output_data));

  for (uint32_t i = 0; i < warmup_runs; ++i) {
    if (ai_llieai_run(ai_network, ai_input, ai_output) != 1) {
      ai_error err = ai_llieai_get_error(ai_network);
      printf("ai warm err %d/%d\r\n", err.type, err.code);
      return;
    }
  }

  for (uint32_t i = 0; i < measure_runs; ++i) {
    uint32_t start_cycles = dwt_cycles();
    ai_i32 batch = ai_llieai_run(ai_network, ai_input, ai_output);
    uint32_t elapsed_cycles = dwt_cycles() - start_cycles;

    if (batch != 1) {
      ai_error err = ai_llieai_get_error(ai_network);
      printf("ai run err %d/%d\r\n", err.type, err.code);
      return;
    }

    if (elapsed_cycles < min_cycles) {
      min_cycles = elapsed_cycles;
    }
    if (elapsed_cycles > max_cycles) {
      max_cycles = elapsed_cycles;
    }
    sum_cycles += elapsed_cycles;
  }

  uint32_t avg_cycles = (uint32_t)(sum_cycles / measure_runs);
  uint32_t cycles_per_us = SystemCoreClock / 1000000U;
  uint32_t avg_us = avg_cycles / cycles_per_us;
  uint32_t min_us = min_cycles / cycles_per_us;
  uint32_t max_us = max_cycles / cycles_per_us;
  uint32_t fps_x100 = (avg_us > 0U) ? (100000000U / avg_us) : 0U;

  printf("\r\nmodel bench\r\n");
  printf("sys=%lu\r\n", SystemCoreClock);
  printf("n=%lu w=%lu\r\n", measure_runs, warmup_runs);
  printf("min=%lu/%lu\r\n", min_cycles, min_us);
  printf("avg=%lu/%lu\r\n", avg_cycles, avg_us);
  printf("max=%lu/%lu\r\n", max_cycles, max_us);
  printf("fps=%lu.%02lu\r\n", fps_x100 / 100U, fps_x100 % 100U);
  printf("out=%u,%u,%u,%u\r\n",
      ai_output_data[0], ai_output_data[1], ai_output_data[2], ai_output_data[3]);
}

static uint32_t elapsed_ms(uint32_t start_ms)
{
  return HAL_GetTick() - start_ms;
}

static void key_finder_config_port(GPIO_TypeDef *port, uint16_t mask)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  GPIO_InitStruct.Pin = mask;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(port, &GPIO_InitStruct);
}

static void key_finder_print_changes(const char *port_name, uint16_t changed, uint16_t state)
{
  for (uint32_t pin = 0; pin < 16U; ++pin) {
    uint16_t mask = (uint16_t)(1U << pin);
    if ((changed & mask) != 0U) {
      printf("keyscan,%s%lu,%lu\r\n",
          port_name,
          (unsigned long)pin,
          (unsigned long)(((state & mask) != 0U) ? 1U : 0U));
    }
  }
}

static void run_key_finder_only(void)
{
  const uint16_t pa_mask = (uint16_t)(0xFFFFU & ~(GPIO_PIN_13 | GPIO_PIN_14));
  const uint16_t pb_mask = (uint16_t)(0xFFFFU & ~(GPIO_PIN_14 | GPIO_PIN_15));
  const uint16_t pc_mask = 0xFFFFU;
  const uint16_t pd_mask = 0xFFFFU;
  const uint16_t pe_mask = 0xFFFFU;
  uint16_t last_pa;
  uint16_t last_pb;
  uint16_t last_pc;
  uint16_t last_pd;
  uint16_t last_pe;

  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();

  key_finder_config_port(GPIOA, pa_mask);
  key_finder_config_port(GPIOB, pb_mask);
  key_finder_config_port(GPIOC, pc_mask);
  key_finder_config_port(GPIOD, pd_mask);
  key_finder_config_port(GPIOE, pe_mask);

  last_pa = (uint16_t)(GPIOA->IDR & pa_mask);
  last_pb = (uint16_t)(GPIOB->IDR & pb_mask);
  last_pc = (uint16_t)(GPIOC->IDR & pc_mask);
  last_pd = (uint16_t)(GPIOD->IDR & pd_mask);
  last_pe = (uint16_t)(GPIOE->IDR & pe_mask);

  printf("keyscan_start,pull=up,pa=0x%04x,pb=0x%04x,pc=0x%04x,pd=0x%04x,pe=0x%04x\r\n",
      last_pa, last_pb, last_pc, last_pd, last_pe);

  while (1) {
    uint16_t state;
    uint16_t changed;

    state = (uint16_t)(GPIOA->IDR & pa_mask);
    changed = (uint16_t)((state ^ last_pa) & pa_mask);
    if (changed != 0U) key_finder_print_changes("PA", changed, state);
    last_pa = state;

    state = (uint16_t)(GPIOB->IDR & pb_mask);
    changed = (uint16_t)((state ^ last_pb) & pb_mask);
    if (changed != 0U) key_finder_print_changes("PB", changed, state);
    last_pb = state;

    state = (uint16_t)(GPIOC->IDR & pc_mask);
    changed = (uint16_t)((state ^ last_pc) & pc_mask);
    if (changed != 0U) key_finder_print_changes("PC", changed, state);
    last_pc = state;

    state = (uint16_t)(GPIOD->IDR & pd_mask);
    changed = (uint16_t)((state ^ last_pd) & pd_mask);
    if (changed != 0U) key_finder_print_changes("PD", changed, state);
    last_pd = state;

    state = (uint16_t)(GPIOE->IDR & pe_mask);
    changed = (uint16_t)((state ^ last_pe) & pe_mask);
    if (changed != 0U) key_finder_print_changes("PE", changed, state);
    last_pe = state;

    HAL_Delay(10);
  }
}

static void configure_key_probe_inputs(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_SYSCFG_CLK_ENABLE();

  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING_FALLING;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;

  GPIO_InitStruct.Pin = GPIO_PIN_0 | GPIO_PIN_1 | GPIO_PIN_2;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = KEY_Pin;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = PE3_Pin;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  HAL_NVIC_SetPriority(EXTI0_IRQn, 6, 0);
  HAL_NVIC_EnableIRQ(EXTI0_IRQn);
  HAL_NVIC_SetPriority(EXTI1_IRQn, 6, 0);
  HAL_NVIC_EnableIRQ(EXTI1_IRQn);
  HAL_NVIC_SetPriority(EXTI2_IRQn, 6, 0);
  HAL_NVIC_EnableIRQ(EXTI2_IRQn);
  HAL_NVIC_SetPriority(EXTI3_IRQn, 6, 0);
  HAL_NVIC_EnableIRQ(EXTI3_IRQn);
  HAL_NVIC_SetPriority(EXTI15_10_IRQn, 6, 0);
  HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);

  key_last_pa2_state = HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_2);
  key_last_pc13_state = HAL_GPIO_ReadPin(KEY_GPIO_Port, KEY_Pin);
  key_last_pe3_state = HAL_GPIO_ReadPin(PE3_GPIO_Port, PE3_Pin);
  printf("button_init,pa2=%lu,pc13=%lu,pe3=%lu\r\n",
      (unsigned long)key_last_pa2_state,
      (unsigned long)key_last_pc13_state,
      (unsigned long)key_last_pe3_state);
}

static void print_key_probe_changes(const char *port_name, uint16_t changed, uint16_t state)
{
  for (uint32_t pin = 0; pin < 16U; ++pin) {
    uint16_t mask = (uint16_t)(1U << pin);
    if ((changed & mask) != 0U) {
      printf("key_probe,%s%lu,%lu\r\n",
          port_name,
          (unsigned long)pin,
          (unsigned long)(((state & mask) != 0U) ? 1U : 0U));
    }
  }
}

static void poll_key_probe(void)
{
  (void)key_probe_last;
  (void)key_probe_last_pa;
  (void)key_probe_last_pb;
  (void)key_probe_last_pc;
  (void)key_probe_last_pd;
  (void)key_probe_last_pe;
}

static void poll_k1_display_toggle(void)
{
  GPIO_PinState pa2_state = HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_2);
  GPIO_PinState pc13_state = HAL_GPIO_ReadPin(KEY_GPIO_Port, KEY_Pin);
  GPIO_PinState pe3_state = HAL_GPIO_ReadPin(PE3_GPIO_Port, PE3_Pin);
  uint32_t now = HAL_GetTick();

  if ((key_irq_pending != 0U) &&
      ((now - key_last_ms) > 250U)) {
    key_irq_pending = 0U;
    if (display_mode == DISPLAY_MODE_MODEL_OUTPUT) {
      display_mode = DISPLAY_MODE_MODEL_INPUT;
      display_ab_mode = 1U;
    } else {
      display_mode = DISPLAY_MODE_MODEL_OUTPUT;
      display_ab_mode = 0U;
    }
    key_last_ms = now;
    printf("button_toggle,pin=0x%04lx,pa2=%lu,pc13=%lu,pe3=%lu,m=%lu\r\n",
        (unsigned long)key_irq_pin,
        (unsigned long)pa2_state,
        (unsigned long)pc13_state,
        (unsigned long)pe3_state,
        (unsigned long)display_mode);
  }

  key_last_pa2_state = pa2_state;
  key_last_pc13_state = pc13_state;
  key_last_pe3_state = pe3_state;
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
  if ((GPIO_Pin == GPIO_PIN_0) || (GPIO_Pin == GPIO_PIN_1) ||
      (GPIO_Pin == GPIO_PIN_2) || (GPIO_Pin == KEY_Pin) ||
      (GPIO_Pin == PE3_Pin)) {
    key_irq_pin = GPIO_Pin;
    key_irq_pending = 1U;
  }
}

static void invalidate_camera_frame(void)
{
  uint32_t addr = (uint32_t)camera_proc_buf;
  uint32_t size = FRAME_WIDTH * FRAME_HEIGHT * 2U;
  uint32_t aligned_addr = addr & ~31UL;
  uint32_t aligned_size = size + (addr - aligned_addr) + 31UL;
  aligned_size &= ~31UL;
  SCB_InvalidateDCache_by_Addr((uint32_t *)aligned_addr, (int32_t)aligned_size);
}

static void clean_dcache_range(const void *ptr, uint32_t size)
{
  uint32_t addr = (uint32_t)ptr;
  uint32_t aligned_addr = addr & ~31UL;
  uint32_t aligned_size = size + (addr - aligned_addr) + 31UL;
  aligned_size &= ~31UL;
  SCB_CleanDCache_by_Addr((uint32_t *)aligned_addr, (int32_t)aligned_size);
}

static void invalidate_dcache_range(const void *ptr, uint32_t size)
{
  uint32_t addr = (uint32_t)ptr;
  uint32_t aligned_addr = addr & ~31UL;
  uint32_t aligned_size = size + (addr - aligned_addr) + 31UL;
  aligned_size &= ~31UL;
  SCB_InvalidateDCache_by_Addr((uint32_t *)aligned_addr, (int32_t)aligned_size);
}

static uint16_t camera_get_pixel(uint32_t x, uint32_t y)
{
  const uint16_t *pixels = (const uint16_t *)camera_proc_buf;
  return pixels[(y * FRAME_WIDTH) + x];
}

static uint16_t camera_get_rgb565_for_ai_variant(uint32_t x, uint32_t y, uint32_t byte_swap, uint32_t rb_swap)
{
  uint16_t p = camera_get_pixel(x, y);

  if (byte_swap != 0U) {
    p = (uint16_t)((p << 8) | (p >> 8));
  }
  if (rb_swap != 0U) {
    p = (uint16_t)((p & 0x07E0U) | ((p & 0x001FU) << 11) | ((p & 0xF800U) >> 11));
  }

  return p;
}

static uint16_t camera_get_rgb565_for_ai(uint32_t x, uint32_t y)
{
  return camera_get_rgb565_for_ai_variant(x, y, CAMERA_RGB565_BYTE_SWAP_FOR_AI, CAMERA_RGB565_RB_SWAP_FOR_AI);
}

static void update_camera_raw_stats(void)
{
  const uint16_t *pixels = (const uint16_t *)camera_proc_buf;
  uint32_t min_value = 0xFFFFU;
  uint32_t max_value = 0U;

  for (uint32_t i = 0; i < (FRAME_WIDTH * FRAME_HEIGHT); ++i) {
    uint32_t p = pixels[i];
    if (p < min_value) min_value = p;
    if (p > max_value) max_value = p;
  }

  camera_raw_min = min_value;
  camera_raw_max = max_value;
}

static void convert_camera_raw_to_display(void)
{
  uint32_t dst = 0;

  for (uint32_t y = 0; y < MODEL_DISPLAY_HEIGHT; ++y) {
    uint32_t sy = (y * FRAME_HEIGHT) / MODEL_DISPLAY_HEIGHT;
    for (uint32_t x = 0; x < MODEL_DISPLAY_WIDTH; ++x) {
      uint32_t sx = (x * FRAME_WIDTH) / MODEL_DISPLAY_WIDTH;
      lcd_back_buf[dst++] = camera_get_pixel(sx, sy);
    }
  }
}

static void preprocess_camera_to_ai_input_variant(uint32_t byte_swap, uint32_t rb_swap)
{
  uint32_t min_value = 255U;
  uint32_t max_value = 0U;

  for (uint32_t y = 0; y < MODEL_HEIGHT; ++y) {
    uint32_t sy = (y * FRAME_HEIGHT) / MODEL_HEIGHT;
    for (uint32_t x = 0; x < MODEL_WIDTH; ++x) {
      uint32_t sx = (x * FRAME_WIDTH) / MODEL_WIDTH;
      uint16_t p = camera_get_rgb565_for_ai_variant(sx, sy, byte_swap, rb_swap);
      uint32_t input_dst = model_io_index(x, y, 0U, ai_input_layout);
      uint32_t shadow_dst = MODEL_INPUT_HWC_INDEX(x, y, 0U);
      ai_u8 r = (ai_u8)(((p >> 11) & 0x1FU) << 3);
      ai_u8 g = (ai_u8)(((p >> 5) & 0x3FU) << 2);
      ai_u8 b = (ai_u8)((p & 0x1FU) << 3);

      ai_runtime_input[input_dst] = r;
      ai_runtime_input[model_io_index(x, y, 1U, ai_input_layout)] = g;
      ai_runtime_input[model_io_index(x, y, 2U, ai_input_layout)] = b;
      model_input_shadow[shadow_dst] = r;
      model_input_shadow[MODEL_INPUT_HWC_INDEX(x, y, 1U)] = g;
      model_input_shadow[MODEL_INPUT_HWC_INDEX(x, y, 2U)] = b;

      if (r < min_value) min_value = r;
      if (g < min_value) min_value = g;
      if (b < min_value) min_value = b;
      if (r > max_value) max_value = r;
      if (g > max_value) max_value = g;
      if (b > max_value) max_value = b;
    }
  }

  model_input_min = min_value;
  model_input_max = max_value;
  act279_min = byte_swap;
  act279_max = rb_swap;
}

static void preprocess_camera_to_ai_input(void)
{
  preprocess_camera_to_ai_input_variant(CAMERA_RGB565_BYTE_SWAP_FOR_AI, CAMERA_RGB565_RB_SWAP_FOR_AI);
}

static void make_lcd_test_pattern(void)
{
  uint32_t dst = 0;
  const uint16_t colors[6] = {
    rgb888_to_rgb565(255U, 0U, 0U),
    rgb888_to_rgb565(0U, 255U, 0U),
    rgb888_to_rgb565(0U, 0U, 255U),
    rgb888_to_rgb565(255U, 255U, 0U),
    rgb888_to_rgb565(0U, 255U, 255U),
    rgb888_to_rgb565(255U, 255U, 255U)
  };

  for (uint32_t y = 0; y < MODEL_DISPLAY_HEIGHT; ++y) {
    for (uint32_t x = 0; x < MODEL_DISPLAY_WIDTH; ++x) {
      lcd_back_buf[dst++] = colors[(x * 6U) / MODEL_DISPLAY_WIDTH];
    }
  }
}

static void fill_constant_model_input(ai_u8 value)
{
  for (uint32_t i = 0; i < AI_LLIEAI_IN_1_SIZE_BYTES; ++i) {
    ai_runtime_input[i] = value;
    model_input_shadow[i] = value;
  }

  model_input_min = value;
  model_input_max = value;
}

static uint16_t rgb888_to_rgb565(uint8_t r, uint8_t g, uint8_t b)
{
  uint16_t p = (uint16_t)(((uint16_t)(r & 0xF8U) << 8) |
                          ((uint16_t)(g & 0xFCU) << 3) |
                          ((uint16_t)b >> 3));

#if LCD_RGB565_STORE_BIG_ENDIAN
  p = (uint16_t)((p << 8) | (p >> 8));
#endif

  return p;
}

#if ENABLE_LCD_RGB565_DUMP
static void dump_lcd_rgb565_frame(const volatile uint16_t *buffer)
{
  printf("lcd_dump_begin,w=%lu,h=%lu,bytes=%lu\r\n",
      (unsigned long)MODEL_DISPLAY_WIDTH,
      (unsigned long)MODEL_DISPLAY_HEIGHT,
      (unsigned long)(MODEL_DISPLAY_WIDTH * MODEL_DISPLAY_HEIGHT * 2U));

  for (uint32_t i = 0; i < (MODEL_DISPLAY_WIDTH * MODEL_DISPLAY_HEIGHT); ++i) {
    if ((i % 16U) == 0U) {
      printf("lcd_dump_data,%lu,", i * 2U);
    }

    uint16_t p = buffer[i];
    printf("%02lX%02lX", (unsigned long)(p & 0xFFU), (unsigned long)(p >> 8));

    if (((i % 16U) == 15U) || (i == ((MODEL_DISPLAY_WIDTH * MODEL_DISPLAY_HEIGHT) - 1U))) {
      printf("\r\n");
    }
  }

  printf("lcd_dump_end\r\n");
}
#endif

static void show_model_display(void)
{
  uint32_t x0 = (ST7735Ctx.Width - MODEL_DISPLAY_WIDTH) / 2U;

  /* Clean the back buffer we just rendered to */
  clean_dcache_range((void*)lcd_back_buf, MODEL_DISPLAY_WIDTH * MODEL_DISPLAY_HEIGHT * 2U);

  /* Swap the LCD buffers */
  volatile uint16_t* temp_lcd = lcd_front_buf;
  lcd_front_buf = lcd_back_buf;
  lcd_back_buf = temp_lcd;

  if (x0 > 0U) {
    ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, x0, MODEL_DISPLAY_HEIGHT, BLACK);
    ST7735_LCD_Driver.FillRect(&st7735_pObj,
                               x0 + MODEL_DISPLAY_WIDTH,
                               0,
                               ST7735Ctx.Width - (x0 + MODEL_DISPLAY_WIDTH),
                               MODEL_DISPLAY_HEIGHT,
                               BLACK);
  }

  /* Transmit the new front buffer */
  ST7735_FillRGBRect(&st7735_pObj,
                     x0,
                     0,
                     (uint8_t *)lcd_front_buf,
                     MODEL_DISPLAY_WIDTH,
                     MODEL_DISPLAY_HEIGHT);
}

static void draw_lcd_diagnostic_overlay(void)
{
  for (uint32_t y = 0; y < MODEL_DISPLAY_HEIGHT; ++y) {
    for (uint32_t x = 0; x < MODEL_DISPLAY_WIDTH; ++x) {
      uint32_t tile = ((x / 12U) + (y / 10U)) & 1U;
      lcd_back_buf[(y * MODEL_DISPLAY_WIDTH) + x] = (tile == 0U) ? MAGENTA : CYAN;
    }
  }
}

static uint32_t model_io_index(uint32_t x, uint32_t y, uint32_t c, uint32_t layout)
{
  if (layout == 1U) {
    return MODEL_IO_CHW_YX_INDEX(x, y, c);
  }
  if (layout == 2U) {
    return MODEL_IO_CHW_XY_INDEX(x, y, c);
  }
  if (layout == 3U) {
    return MODEL_IO_PUBLIC_YCX_INDEX(x, y, c);
  }
  return MODEL_IO_HWC_INDEX(x, y, c);
}

static uint32_t __attribute__((unused)) sample_u8_avg3x3(const ai_u8 *buffer, uint32_t x, uint32_t y, uint32_t c, uint32_t layout)
{
  uint32_t sum = 0;
  uint32_t count = 0;

  for (int32_t dy = -1; dy <= 1; ++dy) {
    int32_t yy = (int32_t)y + dy;
    if ((yy < 0) || (yy >= (int32_t)MODEL_HEIGHT)) {
      continue;
    }
    for (int32_t dx = -1; dx <= 1; ++dx) {
      int32_t xx = (int32_t)x + dx;
      if ((xx < 0) || (xx >= (int32_t)MODEL_WIDTH)) {
        continue;
      }
      sum += buffer[model_io_index((uint32_t)xx, (uint32_t)yy, c, layout)];
      ++count;
    }
  }

  return (count > 0U) ? ((sum + (count / 2U)) / count) : 0U;
}

static uint32_t sample_gain_havg7(const ai_u8 *buffer, uint32_t x, uint32_t y, uint32_t layout)
{
  uint32_t count = 0;
  uint8_t values[9];

  for (int32_t dx = -4; dx <= 4; ++dx) {
    int32_t xx = (int32_t)x + dx;
    if ((xx < 0) || (xx >= (int32_t)MODEL_WIDTH)) {
      continue;
    }
    values[count++] = (uint8_t)((buffer[model_io_index((uint32_t)xx, y, 0U, layout)] +
                                 buffer[model_io_index((uint32_t)xx, y, 1U, layout)] +
                                 buffer[model_io_index((uint32_t)xx, y, 2U, layout)] + 1U) / 3U);
  }

  for (uint32_t i = 1U; i < count; ++i) {
    uint8_t value = values[i];
    uint32_t j = i;
    while ((j > 0U) && (values[j - 1U] > value)) {
      values[j] = values[j - 1U];
      --j;
    }
    values[j] = value;
  }

  return values[count / 2U];
}

static void measure_u8_buffer(const ai_u8 *buffer, uint32_t length, volatile uint32_t *min_out, volatile uint32_t *max_out)
{
  uint32_t min_value = 255U;
  uint32_t max_value = 0U;

  for (uint32_t i = 0; i < length; ++i) {
    uint32_t value = buffer[i];
    if (value < min_value) min_value = value;
    if (value > max_value) max_value = value;
  }

  *min_out = min_value;
  *max_out = max_value;
}

#if ENABLE_AI_OUTPUT_DEBUG
static void tensor_stats_u8(const ai_u8 *buffer, uint32_t length, TensorStats *stats)
{
  uint32_t min_value = 255U;
  uint32_t max_value = 0U;
  uint32_t checksum = 2166136261UL;

  for (uint32_t i = 0; i < length; ++i) {
    uint32_t value = buffer[i];
    if (value < min_value) min_value = value;
    if (value > max_value) max_value = value;
    checksum ^= value;
    checksum *= 16777619UL;
  }

  stats->min_value = min_value;
  stats->max_value = max_value;
  stats->checksum = checksum;
}

static uint32_t checksum_u16(const uint16_t *buffer, uint32_t length)
{
  uint32_t checksum = 2166136261UL;

  for (uint32_t i = 0; i < length; ++i) {
    uint32_t value = buffer[i];
    checksum ^= (value & 0xFFU);
    checksum *= 16777619UL;
    checksum ^= (value >> 8);
    checksum *= 16777619UL;
  }

  return checksum;
}
#endif

#if ENABLE_E2E_DEBUG_STATS || ENABLE_LEGACY_DEBUG_DISPLAY_MODES
static void update_ai_output_candidates_stats(void)
{
  measure_u8_buffer(&ai_activations_d1[0], AI_LLIEAI_OUT_1_SIZE_BYTES, &out0_min, &out0_max);
  measure_u8_buffer(&ai_activations_d1[LLIEAI_OUTPUT_CANDIDATE_OFFSET], AI_LLIEAI_OUT_1_SIZE_BYTES, &out55296_min, &out55296_max);
  measure_u8_buffer(&ai_activations_d1[LLIEAI_OUTPUT_CANDIDATE2_OFFSET], AI_LLIEAI_OUT_1_SIZE_BYTES, &out27648_min, &out27648_max);
  measure_u8_buffer(&ai_activations_d1[LLIEAI_OUTPUT_CANDIDATE3_OFFSET], AI_LLIEAI_OUT_1_SIZE_BYTES, &out55376_min, &out55376_max);
  measure_u8_buffer(&ai_activations_d1[LLIEAI_OUTPUT_CANDIDATE4_OFFSET], AI_LLIEAI_OUT_1_SIZE_BYTES, &out165888_min, &out165888_max);
}
#endif

static void update_io_probe_stats(void)
{
  measure_u8_buffer(ai_runtime_output, AI_LLIEAI_OUT_1_SIZE_BYTES, &model_output_min, &model_output_max);
#if AI_LLIEAI_OUT_NUM >= 2
  measure_u8_buffer(ai_runtime_residual, AI_LLIEAI_OUT_2_SIZE_BYTES, &model_output1_min, &model_output1_max);
#else
  model_output1_min = 0;
  model_output1_max = 0;
#endif
  if (display_mode != DISPLAY_MODE_MODEL_OUTPUT) {
    measure_u8_buffer(ai_runtime_input, AI_LLIEAI_IN_1_SIZE_BYTES, &act279_min, &act279_max);
  }
}

#if ENABLE_AI_OUTPUT_DEBUG
static void debug_capture_ai_output(void)
{
  tensor_stats_u8(ai_runtime_output, AI_LLIEAI_OUT_1_SIZE_BYTES, &debug_out0_stats);
#if AI_LLIEAI_OUT_NUM >= 2
  tensor_stats_u8(ai_runtime_residual, AI_LLIEAI_OUT_2_SIZE_BYTES, &debug_out1_stats);
#endif
}

static void debug_capture_lcd_post(void)
{
  debug_lcd_post_checksum = checksum_u16((const uint16_t *)lcd_back_buf,
                                         MODEL_DISPLAY_WIDTH * MODEL_DISPLAY_HEIGHT);
}

static void debug_record_fixed_input_run(void)
{
  if (display_mode != DISPLAY_MODE_CONSTANT_INPUT_OUTPUT) {
    return;
  }

  if (debug_fixed_runs == 0U) {
    debug_out0_ref = debug_out0_stats.checksum;
    debug_out1_ref = debug_out1_stats.checksum;
    debug_lcd_ref = debug_lcd_post_checksum;
    debug_out0_mismatch = 0U;
    debug_out1_mismatch = 0U;
    debug_lcd_mismatch = 0U;
  } else {
    if (debug_out0_stats.checksum != debug_out0_ref) ++debug_out0_mismatch;
    if (debug_out1_stats.checksum != debug_out1_ref) ++debug_out1_mismatch;
    if (debug_lcd_post_checksum != debug_lcd_ref) ++debug_lcd_mismatch;
  }

  ++debug_fixed_runs;
  if (debug_fixed_runs >= 10U) {
    printf("dbg,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu\r\n",
        debug_out0_stats.checksum,
        debug_out0_stats.min_value,
        debug_out0_stats.max_value,
        debug_out1_stats.checksum,
        debug_out1_stats.min_value,
        debug_out1_stats.max_value,
        debug_lcd_post_checksum,
        debug_out0_mismatch,
        debug_out1_mismatch,
        debug_lcd_mismatch);
    debug_fixed_runs = 0U;
  }
}
#endif

static void __attribute__((unused)) convert_ai_buffer_to_display(const ai_u8 *output_buffer, uint32_t output_offset)
{
  uint32_t dst = 0;
  uint32_t min_value = 255U;
  uint32_t max_value = 0U;

  for (uint32_t y = 0; y < MODEL_DISPLAY_HEIGHT; ++y) {
    uint32_t sy = y + ((MODEL_HEIGHT - MODEL_DISPLAY_HEIGHT) / 2U);
    for (uint32_t x = 0; x < MODEL_DISPLAY_WIDTH; ++x) {
      ai_u8 r = output_buffer[model_io_index(x, sy, 0U, ai_output_layout)];
      ai_u8 g = output_buffer[model_io_index(x, sy, 1U, ai_output_layout)];
      ai_u8 b = output_buffer[model_io_index(x, sy, 2U, ai_output_layout)];
      lcd_back_buf[dst++] = rgb888_to_rgb565(
          r,
          g,
          b);

      if (r < min_value) min_value = r;
      if (g < min_value) min_value = g;
      if (b < min_value) min_value = b;
      if (r > max_value) max_value = r;
      if (g > max_value) max_value = g;
      if (b > max_value) max_value = b;
    }
  }

  model_output_min = min_value;
  model_output_max = max_value;
  active_output_offset = output_offset;
}

static void compose_ai_tail_to_display(const ai_u8 *gain_buffer,
                                       const ai_u8 *residual_buffer,
                                       uint32_t output_offset)
{
  uint32_t dst = 0;
  uint32_t min_value = 255U;
  uint32_t max_value = 0U;
  uint32_t delta_sum = 0U;
  uint32_t delta_max = 0U;
  uint32_t gain_sum = 0U;
  uint32_t gain_max_used = 0U;
  uint32_t residual_q8 = postprocess_residual_q8;

  if (residual_q8 > 256U) {
    residual_q8 = 256U;
  }

  for (uint32_t y = 0; y < MODEL_DISPLAY_HEIGHT; ++y) {
    uint32_t sy = y + ((MODEL_HEIGHT - MODEL_DISPLAY_HEIGHT) / 2U);
    for (uint32_t x = 0; x < MODEL_DISPLAY_WIDTH; ++x) {
      uint8_t out[MODEL_CHANNELS];

      for (uint32_t c = 0; c < MODEL_CHANNELS; ++c) {
        uint32_t public_src = model_io_index(x, sy, c, ai_output_layout);
        uint32_t input_q = model_input_shadow[MODEL_INPUT_HWC_INDEX(x, sy, c)];
        uint32_t gain_head_q = gain_buffer[public_src];
        uint32_t residual_head_q = residual_buffer[public_src];
        uint32_t gain_mapped_q = (gain_head_q + 256U) >> 1;
        uint32_t input_gain_q = (input_q * gain_mapped_q * 6042U + (1U << 19)) >> 20;
        uint32_t residual_mapped_q = residual_tail_lut[residual_head_q];
        uint32_t residual_term = (residual_mapped_q * 13107U * residual_q8 + (1U << 7)) >> 8;
        uint32_t value = (input_gain_q * 84920U + residual_term + (1U << 15)) >> 16;

        if (value > 255U) value = 255U;
        out[c] = (uint8_t)value;

        if (value < min_value) min_value = value;
        if (value > max_value) max_value = value;
        uint32_t abs_delta = (value > input_q) ? (value - input_q) : (input_q - value);
        delta_sum += abs_delta;
        if (abs_delta > delta_max) {
          delta_max = abs_delta;
        }
        gain_sum += gain_mapped_q;
        if (gain_mapped_q > gain_max_used) {
          gain_max_used = gain_mapped_q;
        }
      }

      lcd_back_buf[dst++] = rgb888_to_rgb565(out[0], out[1], out[2]);
    }
  }

  model_output_min = min_value;
  model_output_max = max_value;
  act110_min = delta_sum / (MODEL_DISPLAY_WIDTH * MODEL_DISPLAY_HEIGHT * MODEL_CHANNELS);
  act110_max = delta_max;
  act222_min = gain_sum / (MODEL_DISPLAY_WIDTH * MODEL_DISPLAY_HEIGHT * MODEL_CHANNELS);
  act222_max = gain_max_used;
  act279_min = 0;
  act279_max = 0;
  active_output_offset = output_offset;
}

static void convert_ai_output_to_display(void)
{
#if AI_LLIEAI_OUT_NUM >= 2 && defined(AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS) && (AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS != 0)
  compose_ai_tail_to_display(ai_runtime_gain, ai_runtime_residual, MODEL_COMPOSED_OUTPUT_OFFSET);
#elif AI_LLIEAI_OUT_NUM >= 2
  compose_ai_tail_to_display(ai_runtime_gain, ai_runtime_residual, MODEL_COMPOSED_OUTPUT_OFFSET);
#else
  convert_ai_buffer_to_display(ai_runtime_output, LLIEAI_OUTPUT_OFFSET);
#endif
}

static void __attribute__((unused)) convert_ai_tail_approx_to_display(void)
{
  uint32_t dst = 0;
  uint32_t min_value = 255U;
  uint32_t max_value = 0U;
  uint32_t residual_q8 = postprocess_residual_q8;

  if (residual_q8 > 256U) {
    residual_q8 = 256U;
  }

  for (uint32_t y = 0; y < MODEL_DISPLAY_HEIGHT; ++y) {
      uint32_t sy = y + ((MODEL_HEIGHT - MODEL_DISPLAY_HEIGHT) / 2U);
    for (uint32_t x = 0; x < MODEL_DISPLAY_WIDTH; ++x) {
      uint32_t src = (sy * MODEL_WIDTH) + x;
      uint8_t out[MODEL_CHANNELS];

      for (uint32_t c = 0; c < MODEL_CHANNELS; ++c) {
        uint32_t plane_src = (c * MODEL_PLANE_SIZE) + src;
        int32_t input_term = (int32_t)ai_runtime_input[plane_src] * 244;
        int32_t residual_term = (((int32_t)ai_runtime_output[plane_src] * 102) - 13056) *
                                (int32_t)residual_q8 / 256;
        int32_t value = (input_term + residual_term + 128) >> 8;

        if (value < 0) value = 0;
        if (value > 255) value = 255;
        out[c] = (uint8_t)value;

        if ((uint32_t)value < min_value) min_value = (uint32_t)value;
        if ((uint32_t)value > max_value) max_value = (uint32_t)value;
      }

      lcd_back_buf[dst++] = rgb888_to_rgb565(out[0], out[1], out[2]);
    }
  }

  model_output_min = min_value;
  model_output_max = max_value;
  active_output_offset = LLIEAI_OUTPUT_OFFSET;
}

#if ENABLE_LEGACY_DEBUG_DISPLAY_MODES
static void convert_ai_candidate_output_to_display(void)
{
  convert_ai_buffer_to_display(&ai_activations[LLIEAI_OUTPUT_CANDIDATE_OFFSET], LLIEAI_OUTPUT_CANDIDATE_OFFSET);
}
#endif

static void convert_ai_input_to_display(void)
{
  uint32_t dst = 0;
  act110_min = 0;
  act110_max = 0;
  act222_min = 0;
  act222_max = 0;

  for (uint32_t y = 0; y < MODEL_DISPLAY_HEIGHT; ++y) {
    uint32_t sy = y + ((MODEL_HEIGHT - MODEL_DISPLAY_HEIGHT) / 2U);
    for (uint32_t x = 0; x < MODEL_DISPLAY_WIDTH; ++x) {
      lcd_back_buf[dst++] = rgb888_to_rgb565(
          model_input_shadow[MODEL_INPUT_HWC_INDEX(x, sy, 0U)],
          model_input_shadow[MODEL_INPUT_HWC_INDEX(x, sy, 1U)],
          model_input_shadow[MODEL_INPUT_HWC_INDEX(x, sy, 2U)]);
    }
  }
}

static int run_llie_once(void)
{
  memcpy(model_input_runtime_snapshot, ai_runtime_input, AI_LLIEAI_IN_1_SIZE_BYTES);
  clean_dcache_range(ai_runtime_input, AI_LLIEAI_IN_1_SIZE_BYTES);
#if defined(AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS) && (AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS != 0) && (AI_LLIEAI_OUT_NUM == 1)
  ai_run_input[0].data = AI_HANDLE_PTR(ai_runtime_input);
  ai_run_output[0].data = AI_HANDLE_PTR(ai_network_output);
  ai_i32 batch = ai_llieai_run(ai_network, ai_run_input, ai_run_output);
  if (batch == 1) {
    memcpy(ai_runtime_output, ai_network_output, AI_LLIEAI_OUT_1_SIZE_BYTES);
  }
#elif (AI_LLIEAI_OUT_NUM >= 2) && defined(AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS) && (AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS != 0)
  ai_run_input[0].data = AI_HANDLE_PTR(ai_runtime_input);
  ai_i32 batch = ai_llieai_forward(ai_network, ai_run_input);
#elif AI_LLIEAI_OUT_NUM >= 2
  ai_run_input[0].data = AI_HANDLE_PTR(ai_runtime_input);
  ai_run_output[0].data = AI_HANDLE_PTR(ai_runtime_gain);
  ai_run_output[1].data = AI_HANDLE_PTR(ai_runtime_residual);
  ai_i32 batch = ai_llieai_run(ai_network, ai_run_input, ai_run_output);
#else
  ai_run_input[0].data = AI_HANDLE_PTR(ai_runtime_input);
  ai_run_output[0].data = AI_HANDLE_PTR(ai_runtime_output);
  memset(ai_runtime_output, 0xA5, AI_LLIEAI_OUT_1_SIZE_BYTES);
  ai_i32 batch = ai_llieai_run(ai_network, ai_run_input, ai_run_output);
#endif
  if (batch != 1) {
    ai_error err = ai_llieai_get_error(ai_network);
    printf("ai run err %d/%d\r\n", err.type, err.code);
    return -1;
  }
  clean_dcache_range(model_input_runtime_snapshot, AI_LLIEAI_IN_1_SIZE_BYTES);
  return 0;
}

#if ENABLE_EXTERNAL_IO_MODE
static int run_llie_external_constant_once(ai_u8 value)
{
  memset(ai_runtime_input, value, AI_LLIEAI_IN_1_SIZE_BYTES);
  memset(model_input_shadow, value, sizeof(model_input_shadow));
  memset(ai_output_data, 0xA5, sizeof(ai_output_data));

  model_input_min = value;
  model_input_max = value;
  external_input_min = value;
  external_input_max = value;

  ai_run_input[0].data = AI_HANDLE_PTR(ai_runtime_input);
#if AI_LLIEAI_OUT_NUM >= 2 && defined(AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS) && (AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS != 0)
  ai_run_output[0].data = AI_HANDLE_PTR(ai_runtime_output);
#else
  ai_run_output[0].data = AI_HANDLE_PTR(ai_output_data);
#endif
#if AI_LLIEAI_OUT_NUM >= 2
  ai_run_output[1].data = AI_HANDLE_PTR(ai_runtime_residual);
#endif
  ai_i32 batch = ai_llieai_run(ai_network, ai_run_input, ai_run_output);

  if (batch != 1) {
    ai_error err = ai_llieai_get_error(ai_network);
    printf("ai ext err %d/%d\r\n", err.type, err.code);
    return -1;
  }

#if AI_LLIEAI_OUT_NUM >= 2 && defined(AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS) && (AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS != 0)
  measure_u8_buffer(ai_runtime_output, AI_LLIEAI_OUT_1_SIZE_BYTES, &external_output_min, &external_output_max);
#else
  measure_u8_buffer(ai_output_data, AI_LLIEAI_OUT_1_SIZE_BYTES, &external_output_min, &external_output_max);
#endif
  return 0;
}
#endif

#if 0
static int run_llie_once_with_outputs(void)
{
  memcpy(model_input_runtime_snapshot, ai_runtime_input, AI_LLIEAI_IN_1_SIZE_BYTES);
  clean_dcache_range(ai_runtime_input, AI_LLIEAI_IN_1_SIZE_BYTES);
  memset(ai_runtime_output, 0xA5, AI_LLIEAI_OUT_1_SIZE_BYTES);
#if AI_LLIEAI_OUT_NUM >= 2
  memset(ai_runtime_residual, 0x5A, AI_LLIEAI_OUT_2_SIZE_BYTES);
#endif
  memset(ai_external_output0, 0xA5, AI_LLIEAI_OUT_1_SIZE_BYTES);
  memset(ai_external_output1, 0x5A, AI_LLIEAI_OUT_2_SIZE_BYTES);
  clean_dcache_range(ai_runtime_output, AI_LLIEAI_OUT_1_SIZE_BYTES);
#if AI_LLIEAI_OUT_NUM >= 2
  clean_dcache_range(ai_runtime_residual, AI_LLIEAI_OUT_2_SIZE_BYTES);
#endif
  clean_dcache_range(ai_external_output0, AI_LLIEAI_OUT_1_SIZE_BYTES);
  clean_dcache_range(ai_external_output1, AI_LLIEAI_OUT_2_SIZE_BYTES);

  ai_run_input[0].data = AI_HANDLE_PTR(ai_runtime_input);
#if AI_LLIEAI_OUT_NUM >= 2
  ai_run_output[0].data = AI_HANDLE_PTR(ai_external_output0);
  ai_run_output[1].data = AI_HANDLE_PTR(ai_external_output1);
#else
  ai_run_output[0].data = AI_HANDLE_PTR(ai_external_output0);
#endif
  ai_i32 batch = ai_llieai_run(ai_network, ai_run_input, ai_run_output);

  if (batch != 1) {
    ai_error err = ai_llieai_get_error(ai_network);
    printf("ai run2 err %d/%d\r\n", err.type, err.code);
    return -1;
  }
  return 0;
}
#endif

static void report_e2e(uint32_t frame_ms)
{
  static uint32_t last_report_ms = 0;
  static uint32_t frame_count = 0;
  static uint32_t sum_ms = 0;

  ++frame_count;
  sum_ms += frame_ms;
  E2E_LastMs = frame_ms;

  if (elapsed_ms(last_report_ms) >= 1000U) {
    E2E_FPS = frame_count;
    total_fps = frame_count;
    E2E_AvgMs = (frame_count > 0U) ? (sum_ms / frame_count) : 0U;

    update_io_probe_stats();
    printf("v=%lu,m=%lu,y=%lu/%lu,g=%lu/%lu,l=%lu,p=%lu,a=%lu,q=%lu,t=%lu,f=%lu,r=%lu/%lu,i=%lu/%lu,o=%lu/%lu,u=%lu/%lu,b=%lu/%lu,c=%lu/%lu,d=%lu/%lu,fb=%lu,io=%lu/%lu,e=%lu\r\n",
        fw_probe_version,
        display_mode,
        ai_input_layout,
        ai_output_layout,
        postprocess_gain_q8,
        postprocess_residual_q8,
        camera_display_ms,
        preprocess_ms,
        inference_ms,
        postprocess_ms,
        total_ms,
        total_fps,
        camera_raw_min,
        camera_raw_max,
        model_input_min,
        model_input_max,
        model_output_min,
        model_output_max,
        model_output1_min,
        model_output1_max,
        act110_min,
        act110_max,
        act222_min,
        act222_max,
        act279_min,
        act279_max,
        display_fallback_count,
        ai_input_runtime_offset,
        ai_output_runtime_offset,
        dcmi_error_count);

    frame_count = 0;
    sum_ms = 0;
    last_report_ms = HAL_GetTick();
  }
}

void HAL_DCMI_FrameEventCallback(DCMI_HandleTypeDef *hdcmi_cb)
{
  (void)hdcmi_cb;
#if ENABLE_PIPELINE_PROFILING
  uint32_t now_cycles = dwt_cycles();
  if (profile_camera_dma_mark_cycles != 0U) {
    profile_camera_dma_cycles = now_cycles - profile_camera_dma_mark_cycles;
    profile_camera_dma_valid = 1U;
  }
  profile_camera_dma_mark_cycles = now_cycles;
#endif
  dcmi_frame_ready = 1;
}

void HAL_DCMI_ErrorCallback(DCMI_HandleTypeDef *hdcmi_cb)
{
  (void)hdcmi_cb;
  ++dcmi_error_count;
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  boot_stage = 1;

  /* USER CODE END 1 */

  /* MPU Configuration--------------------------------------------------------*/
  MPU_Config();

  /* Enable the CPU Cache */

  /* Enable I-Cache---------------------------------------------------------*/
  SCB_EnableICache();

  /* Enable D-Cache---------------------------------------------------------*/
#if !TEST_CUBEAI_FIXED_INPUT_ONLY
  SCB_EnableDCache();
#endif

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */
  __HAL_RCC_D2SRAM1_CLK_ENABLE();
  __HAL_RCC_D2SRAM2_CLK_ENABLE();
  __HAL_RCC_D2SRAM3_CLK_ENABLE();
  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_CRC_Init();
  MX_USART1_UART_Init();
  /* USER CODE BEGIN 2 */
  dwt_counter_init();
  HAL_Delay(100);
#if TEST_KEY_FINDER_ONLY
  run_key_finder_only();
#endif
  configure_key_probe_inputs();
  key_last_pc13_state = HAL_GPIO_ReadPin(KEY_GPIO_Port, KEY_Pin);
  key_last_pe3_state = HAL_GPIO_ReadPin(PE3_GPIO_Port, PE3_Pin);
  key_last_ms = HAL_GetTick();

#if TEST_CUBEAI_FIXED_INPUT_ONLY
  boot_stage = 31;
  ai_input_layout = 1U;
  ai_output_layout = 1U;
  if (llie_ai_init() != 0) {
    boot_stage = 501;
    Error_Handler();
  }
  print_model_identity();
  fill_constant_model_input(32U);
  boot_stage = 32;
  if (run_llie_once() != 0) {
    boot_stage = 502;
    Error_Handler();
  }
#if ENABLE_TENSOR_EQUIV_DUMP
  dump_tensor_equivalence_frame();
#endif
  boot_stage = 40;
  while (1) {
    HAL_Delay(1000);
  }
#endif

  {
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    __HAL_RCC_GPIOE_CLK_ENABLE();

    HAL_GPIO_WritePin(GPIOE, LCD_CS_Pin | LCD_WR_RS_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_10, GPIO_PIN_RESET);

    GPIO_InitStruct.Pin = LCD_CS_Pin | LCD_WR_RS_Pin | GPIO_PIN_10;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);
  }
  MX_SPI4_Init();
  boot_stage = 20;
#if TEST_LCD_EARLY_ONLY
  MX_TIM1_Init();
  LCD_Test();
  LCD_SetBrightness(999U);
  {
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    HAL_TIMEx_PWMN_Stop(&htim1, TIM_CHANNEL_2);
    __HAL_RCC_GPIOE_CLK_ENABLE();
    GPIO_InitStruct.Pin = GPIO_PIN_10;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);
    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_10, GPIO_PIN_RESET);
  }
  printf("lcd early diag fw=%lu,w=%lu,h=%lu\r\n",
      fw_probe_version,
      (uint32_t)ST7735Ctx.Width,
      (uint32_t)ST7735Ctx.Height);
  while (1) {
    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_10, GPIO_PIN_RESET);
    boot_stage = 201;
    ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, RED);
    HAL_Delay(500);
    boot_stage = 202;
    ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, GREEN);
    HAL_Delay(500);
    boot_stage = 203;
    ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, BLUE);
    HAL_Delay(500);
    boot_stage = 204;
    ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, WHITE);
    HAL_Delay(500);
  }
#endif

#if TEST_LCD_PATTERN
  {
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    display_mode = DISPLAY_MODE_LCD_TEST;
    camera_dma_enabled = 0U;
    HAL_TIMEx_PWMN_Stop(&htim1, TIM_CHANNEL_2);
    __HAL_RCC_GPIOE_CLK_ENABLE();
    GPIO_InitStruct.Pin = GPIO_PIN_10;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

    while (1) {
      boot_stage = 80;
      HAL_GPIO_WritePin(GPIOE, GPIO_PIN_10, GPIO_PIN_RESET);
      ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, RED);
      HAL_Delay(500);

      boot_stage = 81;
      HAL_GPIO_WritePin(GPIOE, GPIO_PIN_10, GPIO_PIN_SET);
      ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, GREEN);
      HAL_Delay(500);

      boot_stage = 82;
      ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, BLUE);
      HAL_Delay(500);

      boot_stage = 83;
      ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, WHITE);
      HAL_Delay(500);
    }
  }
#else
  MX_CRC_Init();
  boot_stage = 21;
  MX_USART1_UART_Init();
  boot_stage = 22;
  MX_DMA_Init();
  boot_stage = 23;
  MX_DCMI_Init();
  boot_stage = 24;
  MX_I2C1_Init();
  boot_stage = 25;

#if TEST_CAMERA_ID_ONLY
  hcamera.hi2c = &hi2c1;
  hcamera.addr = OV5640_ADDRESS;
  hcamera.timeout = 100U;

  boot_stage = 260;
  Camera_XCLK_Set(XCLK_TIM);
  HAL_Delay(300);
  for (uint32_t cam_try = 0; cam_try < 5U; ++cam_try) {
    HAL_Delay(100);
    Camera_read_id(&hcamera);
    camera_tim_detected_id = hcamera.device_id;
    if (hcamera.device_id == 0x5640U) {
      break;
    }
  }

  boot_stage = 261;
  Camera_XCLK_Set(XCLK_MCO);
  HAL_Delay(300);
  for (uint32_t cam_try = 0; cam_try < 5U; ++cam_try) {
    HAL_Delay(100);
    Camera_read_id(&hcamera);
    camera_mco_detected_id = hcamera.device_id;
    if (hcamera.device_id == 0x5640U) {
      break;
    }
  }

  camera_detected_id = (camera_tim_detected_id == 0x5640U) ?
      camera_tim_detected_id : camera_mco_detected_id;

  MX_TIM1_Init();
  MX_SPI4_Init();
  LCD_Test();
  {
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    HAL_TIMEx_PWMN_Stop(&htim1, TIM_CHANNEL_2);
    __HAL_RCC_GPIOE_CLK_ENABLE();
    GPIO_InitStruct.Pin = GPIO_PIN_10;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);
    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_10, GPIO_PIN_RESET);
  }

  if (camera_tim_detected_id == 0x5640U) {
    boot_stage = 262;
    ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, GREEN);
  } else if (camera_mco_detected_id == 0x5640U) {
    boot_stage = 263;
    ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, BLUE);
  } else {
    boot_stage = 264;
    ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, RED);
  }

  while (1) {
    HAL_Delay(1000);
  }
#endif

  HAL_TIMEx_PWMN_Stop(&htim1, TIM_CHANNEL_2);
  Camera_XCLK_Set(XCLK_MCO);
  {
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    __HAL_RCC_GPIOE_CLK_ENABLE();
    GPIO_InitStruct.Pin = GPIO_PIN_10;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);
    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_10, GPIO_PIN_RESET);
  }
#if TEST_LCD_AFTER_XCLK_ONLY
  display_mode = DISPLAY_MODE_LCD_TEST;
  printf("lcd diag after xclk fw=%lu\r\n", fw_probe_version);
  while (1) {
    ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, RED);
    HAL_Delay(500);
    ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, GREEN);
    HAL_Delay(500);
    ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, BLUE);
    HAL_Delay(500);
    ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, WHITE);
    HAL_Delay(500);
  }
#endif
  for (uint32_t cam_try = 0; cam_try < 5U; ++cam_try) {
    HAL_Delay(200);
    Camera_Init_Device(&hi2c1, FRAMESIZE_QQVGA);
    printf("cam try=%lu id=0x%04x\r\n", cam_try, hcamera.device_id);
    if (hcamera.device_id == 0x5640U) {
      break;
    }
  }
  camera_detected_id = hcamera.device_id;
  boot_stage = 30;
  if (hcamera.device_id != 0x5640U) {
    boot_stage = 401;
    printf("cam id=0x%04x\r\n", hcamera.device_id);
    Error_Handler();
  }

  LCD_Test();
  {
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    HAL_TIMEx_PWMN_Stop(&htim1, TIM_CHANNEL_2);
    __HAL_RCC_GPIOE_CLK_ENABLE();
    GPIO_InitStruct.Pin = GPIO_PIN_10;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);
    HAL_GPIO_WritePin(GPIOE, GPIO_PIN_10, GPIO_PIN_RESET);
  }
  ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, WHITE);
  HAL_Delay(200);

#if TEST_LCD_RGBRECT_STREAM_ONLY
  printf("lcd_rgbrect_stream_only,w=%lu,h=%lu,be=%lu\r\n",
      (unsigned long)MODEL_DISPLAY_WIDTH,
      (unsigned long)MODEL_DISPLAY_HEIGHT,
      (unsigned long)LCD_RGB565_STORE_BIG_ENDIAN);
  ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, BLACK);
  while (1) {
    make_lcd_test_pattern();
    show_model_display();
    HAL_Delay(500);
  }
#endif

  if (llie_ai_init() != 0) {
    boot_stage = 501;
    ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, MAGENTA);
    Error_Handler();
  }
  print_model_identity();
#if TEST_FIXED_INPUT_AI_TO_LCD
  camera_dma_enabled = 0U;
  display_mode = DISPLAY_MODE_CONSTANT_INPUT_OUTPUT;
  ai_input_layout = 1U;
  ai_output_layout = 1U;
  postprocess_gain_q8 = 256U;
  postprocess_residual_q8 = 0U;
  fill_constant_model_input(32U);
  if (run_llie_once() != 0) {
    ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, RED);
    Error_Handler();
  }
  convert_ai_output_to_display();
  show_model_display();
  printf("fixed_lcd_done,o=%lu/%lu\r\n", model_output_min, model_output_max);
  while (1) {
    HAL_Delay(1000);
  }
#endif
#if TEST_CUBEAI_FIXED_INPUT_ONLY
  camera_dma_enabled = 0U;
  display_mode = DISPLAY_MODE_CONSTANT_INPUT_OUTPUT;
  ai_input_layout = 1U;
  ai_output_layout = 1U;
  postprocess_gain_q8 = 256U;
  postprocess_residual_q8 = 0U;
  fill_constant_model_input(32U);
  if (run_llie_once() != 0) {
    Error_Handler();
  }
#if ENABLE_TENSOR_EQUIV_DUMP
  dump_tensor_equivalence_frame();
#endif
  while (1) {
    HAL_Delay(1000);
  }
#endif
  boot_stage = 40;

#if TEST_LCD_PATTERN || PIPELINE_CORRECTNESS_MODE || TEST_FIXED_INPUT_AI || TEST_FIXED_INPUT_AI_TO_LCD
  display_mode = DISPLAY_MODE_CONSTANT_INPUT_OUTPUT;
#elif TEST_LCD_PATTERN
  display_mode = DISPLAY_MODE_LCD_TEST;
#elif TEST_CAMERA_TO_AI_NO_LCD
  display_mode = DISPLAY_MODE_MODEL_INPUT;
#else
  display_mode = DISPLAY_MODE_MODEL_OUTPUT;
#endif

#if ENABLE_PIPELINE_PROFILING
  uint32_t camera_start_begin_cycles = dwt_cycles();
  profile_camera_dma_mark_cycles = camera_start_begin_cycles;
#endif
#if TEST_LCD_PATTERN || PIPELINE_CORRECTNESS_MODE || TEST_FIXED_INPUT_AI || TEST_FIXED_INPUT_AI_TO_LCD || TEST_BLOCKING_SNAPSHOT_LOOP
  camera_dma_enabled = 0U;
  boot_stage = 60;
#else
  if (HAL_DCMI_Start_DMA(&hdcmi,
                         DCMI_MODE_SNAPSHOT,
                         (uint32_t)camera_dma_buf,
                         (FRAME_WIDTH * FRAME_HEIGHT * 2U) / 4U) != HAL_OK) {
    boot_stage = 601;
    ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, MAGENTA);
    printf("dcmi start fail\r\n");
    Error_Handler();
  }
  camera_dma_enabled = 1U;
#if ENABLE_PIPELINE_PROFILING
  profile_camera_start_overhead_cycles = dwt_cycles() - camera_start_begin_cycles;
#endif
  boot_stage = 60;
#endif
  ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, GREEN);
#endif
#if ENABLE_PIPELINE_PROFILING
  profile_wait_start_cycles = dwt_cycles();
#endif

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    poll_key_probe();
    poll_k1_display_toggle();
#if PIPELINE_CORRECTNESS_MODE || TEST_FIXED_INPUT_AI || TEST_FIXED_INPUT_AI_TO_LCD
    if (dcmi_frame_ready == 0U) {
      HAL_Delay(1);
      dcmi_frame_ready = 1U;
    }
#endif
#if TEST_BLOCKING_SNAPSHOT_LOOP
    if ((dcmi_frame_ready == 0U) && (camera_dma_enabled == 0U)) {
      hdcmi.State = HAL_DCMI_STATE_READY;
      if (HAL_DCMI_Start_DMA(&hdcmi,
                             DCMI_MODE_SNAPSHOT,
                             (uint32_t)camera_dma_buf,
                             (FRAME_WIDTH * FRAME_HEIGHT * 2U) / 4U) != HAL_OK) {
        HAL_DCMI_Stop(&hdcmi);
        if (hdcmi.DMA_Handle != NULL) {
          HAL_DMA_Abort(hdcmi.DMA_Handle);
        }
        hdcmi.State = HAL_DCMI_STATE_READY;
        HAL_Delay(2);
        (void)HAL_DCMI_Start_DMA(&hdcmi,
                                 DCMI_MODE_SNAPSHOT,
                                 (uint32_t)camera_dma_buf,
                                 (FRAME_WIDTH * FRAME_HEIGHT * 2U) / 4U);
      }
      camera_dma_enabled = 1U;
    }
#endif
    if (dcmi_frame_ready != 0U) {
      boot_stage = 70;
      uint32_t frame_start_ms = HAL_GetTick();
      uint32_t start_ms;
      uint32_t frame_camera_dma_enabled = camera_dma_enabled;
#if ENABLE_PIPELINE_PROFILING
      PipelineProfile frame_profile = {0};
      uint32_t frame_start_cycles = dwt_cycles();
      uint32_t profile_step_cycles;

      frame_profile.camera_start_cycles = profile_camera_start_overhead_cycles;
      frame_profile.camera_wait_cycles = frame_start_cycles - profile_wait_start_cycles;
      frame_profile.camera_dma_cycles = (profile_camera_dma_valid != 0U) ? profile_camera_dma_cycles : 0U;
#endif

      dcmi_frame_ready = 0;
#if TEST_FULL_PIPELINE_ONE_SHOT
      if (frame_camera_dma_enabled != 0U) {
        HAL_DCMI_Stop(&hdcmi);
        if (hdcmi.DMA_Handle != NULL) {
          HAL_DMA_Abort(hdcmi.DMA_Handle);
        }
        hdcmi.State = HAL_DCMI_STATE_READY;
        volatile uint16_t* temp_cam = camera_dma_buf;
        camera_dma_buf = camera_proc_buf;
        camera_proc_buf = temp_cam;
        camera_dma_enabled = 0U;
      }
#else
      if (frame_camera_dma_enabled != 0U) {
#if CAMERA_STOP_DURING_PROCESSING
        HAL_DCMI_Stop(&hdcmi);
        if (hdcmi.DMA_Handle != NULL) {
          HAL_DMA_Abort(hdcmi.DMA_Handle);
        }
        hdcmi.State = HAL_DCMI_STATE_READY;
        camera_dma_enabled = 0U;
#endif
        volatile uint16_t* temp_cam = camera_dma_buf;
        camera_dma_buf = camera_proc_buf;
        camera_proc_buf = temp_cam;
      }
#endif
      preprocess_ms = 0;
#if ENABLE_PIPELINE_PROFILING
      frame_profile.input_copy_cycles = 0U;
#endif

      if ((display_mode != DISPLAY_MODE_LCD_TEST) && (frame_camera_dma_enabled != 0U)) {
#if ENABLE_PIPELINE_PROFILING
        profile_step_cycles = dwt_cycles();
#endif
        invalidate_camera_frame();
#if ENABLE_PIPELINE_PROFILING
        frame_profile.camera_cache_cycles = dwt_cycles() - profile_step_cycles;
        profile_step_cycles = dwt_cycles();
#endif
        update_camera_raw_stats();
      }

#if TEST_FULL_PIPELINE_ONE_SHOT && ENABLE_TENSOR_EQUIV_DUMP
      dump_input_decode_variants();
#endif

      if ((display_mode != DISPLAY_MODE_LCD_TEST) &&
          (display_mode != DISPLAY_MODE_CAMERA_RAW)) {
        start_ms = HAL_GetTick();
        preprocess_camera_to_ai_input();
        preprocess_ms = elapsed_ms(start_ms);
#if ENABLE_PIPELINE_PROFILING
        frame_profile.preprocess_cycles = dwt_cycles() - profile_step_cycles;
#endif
      }

      if (display_mode == DISPLAY_MODE_LCD_TEST) {
#if ENABLE_PIPELINE_PROFILING
        profile_step_cycles = dwt_cycles();
#endif
        make_lcd_test_pattern();
        boot_stage = 71;
#if ENABLE_PIPELINE_PROFILING
        frame_profile.rgb565_cycles = dwt_cycles() - profile_step_cycles;
#endif
        uint32_t display_start_ms = HAL_GetTick();
#if ENABLE_PIPELINE_PROFILING
        profile_step_cycles = dwt_cycles();
#endif
        boot_stage = 72;
        show_model_display();
        boot_stage = 73;
#if ENABLE_PIPELINE_PROFILING
        frame_profile.lcd_transfer_cycles = dwt_cycles() - profile_step_cycles;
#endif
        camera_display_ms = elapsed_ms(display_start_ms);
        inference_ms = 0;
        postprocess_ms = 0;
        model_output_min = 0;
        model_output_max = 0;
      } else if (display_mode == DISPLAY_MODE_CAMERA_RAW) {
#if ENABLE_PIPELINE_PROFILING
        profile_step_cycles = dwt_cycles();
#endif
        convert_camera_raw_to_display();
#if ENABLE_PIPELINE_PROFILING
        frame_profile.rgb565_cycles = dwt_cycles() - profile_step_cycles;
#endif
        uint32_t display_start_ms = HAL_GetTick();
#if ENABLE_PIPELINE_PROFILING
        profile_step_cycles = dwt_cycles();
#endif
        show_model_display();
#if ENABLE_PIPELINE_PROFILING
        frame_profile.lcd_transfer_cycles = dwt_cycles() - profile_step_cycles;
#endif
        camera_display_ms = elapsed_ms(display_start_ms);
        inference_ms = 0;
        postprocess_ms = 0;
        model_output_min = 0;
        model_output_max = 0;
      } else {
        if ((display_mode == DISPLAY_MODE_MODEL_OUTPUT) ||
            (display_mode == DISPLAY_MODE_CONSTANT_INPUT_OUTPUT) ||
#if ENABLE_LEGACY_DEBUG_DISPLAY_MODES
            (display_mode == DISPLAY_MODE_CANDIDATE_OUTPUT) ||
#endif
#if ENABLE_EXTERNAL_IO_MODE
            (display_mode == DISPLAY_MODE_EXTERNAL_IO_OUTPUT) ||
#endif
#if ENABLE_LEGACY_DEBUG_DISPLAY_MODES
            (display_mode == DISPLAY_MODE_GAIN_MAP) ||
            (display_mode == DISPLAY_MODE_RESIDUAL_MAP) ||
#endif
            (0U != 0U)) {
          uint32_t infer_start_ms = HAL_GetTick();
#if ENABLE_PIPELINE_PROFILING
          profile_step_cycles = dwt_cycles();
#endif
#if ENABLE_EXTERNAL_IO_MODE
          if (display_mode == DISPLAY_MODE_EXTERNAL_IO_OUTPUT) {
            if (run_llie_external_constant_once(160U) != 0) {
              ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, RED);
              Error_Handler();
            }
          } else
#endif
          {
            if (display_mode == DISPLAY_MODE_CONSTANT_INPUT_OUTPUT) {
              fill_constant_model_input(32U);
            }
            if (run_llie_once() != 0) {
              ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, RED);
              Error_Handler();
            }
          }
#if ENABLE_TENSOR_EQUIV_DUMP
          if ((tensor_dump_request != 0U) && (tensor_dump_done == 0U)) {
            tensor_dump_request = 0U;
            dump_tensor_equivalence_frame();
            tensor_dump_done = 1U;
          }
#endif
#if ENABLE_AI_OUTPUT_DEBUG
          debug_capture_ai_output();
#endif
#if ENABLE_PIPELINE_PROFILING
          frame_profile.inference_cycles = dwt_cycles() - profile_step_cycles;
#endif
          inference_ms = elapsed_ms(infer_start_ms);

          uint32_t postprocess_start_ms = HAL_GetTick();
#if ENABLE_PIPELINE_PROFILING
          profile_step_cycles = dwt_cycles();
#endif
#if ENABLE_E2E_DEBUG_STATS
          update_ai_output_candidates_stats();
#else
#if ENABLE_LEGACY_DEBUG_DISPLAY_MODES
          if (display_mode == DISPLAY_MODE_CANDIDATE_OUTPUT) {
            update_ai_output_candidates_stats();
          }
#endif
#endif
          if (0) {
#if ENABLE_EXTERNAL_IO_MODE
          } else if (display_mode == DISPLAY_MODE_EXTERNAL_IO_OUTPUT) {
#if AI_LLIEAI_OUT_NUM >= 2 && defined(AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS) && (AI_LLIEAI_OUTPUTS_IN_ACTIVATIONS != 0)
            convert_ai_buffer_to_display(ai_runtime_output, LLIEAI_OUTPUT0_ACTIVATION_OFFSET);
#else
            convert_ai_buffer_to_display(ai_output_data, 0xFFFFFFFFU);
#endif
#endif
#if ENABLE_LEGACY_DEBUG_DISPLAY_MODES
          } else if (display_mode == DISPLAY_MODE_CANDIDATE_OUTPUT) {
            convert_ai_candidate_output_to_display();
          } else if (display_mode == DISPLAY_MODE_GAIN_MAP) {
            convert_ai_buffer_to_display(ai_runtime_output, LLIEAI_OUTPUT0_ACTIVATION_OFFSET);
          } else if (display_mode == DISPLAY_MODE_RESIDUAL_MAP) {
            convert_ai_buffer_to_display(ai_runtime_residual, LLIEAI_OUTPUT1_ACTIVATION_OFFSET);
#endif
          } else {
#if TEST_AI_OUTPUT_LAYOUT_SWEEP
            static const uint32_t output_layout_sweep[4] = {1U, 2U, 3U, 4U};
            ai_output_layout = output_layout_sweep[ai_output_layout_sweep_frame % 4U];
            ++ai_output_layout_sweep_frame;
#endif
            convert_ai_output_to_display();
#if DISPLAY_SAFE_INPUT_AFTER_AI
            convert_ai_input_to_display();
#endif
            if (model_output_max <= 2U) {
              ++display_fallback_count;
            }
          }
#if ENABLE_AI_OUTPUT_DEBUG
          debug_capture_lcd_post();
#endif
#if ENABLE_PIPELINE_PROFILING
          frame_profile.postprocess_cycles = dwt_cycles() - profile_step_cycles;
          frame_profile.rgb565_cycles = frame_profile.postprocess_cycles;
#endif
          postprocess_ms = elapsed_ms(postprocess_start_ms);
        } else {
#if ENABLE_PIPELINE_PROFILING
          profile_step_cycles = dwt_cycles();
#endif
          inference_ms = 0;
          postprocess_ms = 0;
          model_output_min = 0;
          model_output_max = 0;
          convert_ai_input_to_display();
#if ENABLE_PIPELINE_PROFILING
          frame_profile.rgb565_cycles = dwt_cycles() - profile_step_cycles;
#endif
        }
        uint32_t display_start_ms = HAL_GetTick();
#if ENABLE_PIPELINE_PROFILING
        profile_step_cycles = dwt_cycles();
#endif
#if TEST_AI_OUTPUT_LAYOUT_SWEEP
        if (display_mode == DISPLAY_MODE_MODEL_OUTPUT) {
          uint32_t marker = GREEN;
          if (ai_output_layout == 1U) {
            marker = RED;
          } else if (ai_output_layout == 2U) {
            marker = BLUE;
          } else if (ai_output_layout == 4U) {
            marker = WHITE;
          }
          ST7735_LCD_Driver.FillRect(&st7735_pObj, 0, 0, ST7735Ctx.Width, ST7735Ctx.Height, marker);
        }
#endif
        show_model_display();
#if ENABLE_AI_OUTPUT_DEBUG
        debug_record_fixed_input_run();
#endif
#if ENABLE_PIPELINE_PROFILING
        frame_profile.lcd_transfer_cycles = dwt_cycles() - profile_step_cycles;
#endif
        camera_display_ms = elapsed_ms(display_start_ms);
      }

      total_ms = elapsed_ms(frame_start_ms);
#if ENABLE_PIPELINE_PROFILING
      frame_profile.total_cycles = dwt_cycles() - frame_start_cycles;
      profile_store_frame(&frame_profile);
      profile_wait_start_cycles = dwt_cycles();
      profile_print_report_once();
#endif
      report_e2e(total_ms);
#if CAMERA_STOP_DURING_PROCESSING && !TEST_FULL_PIPELINE_ONE_SHOT && !TEST_BLOCKING_SNAPSHOT_LOOP
      if (frame_camera_dma_enabled != 0U) {
        dcmi_frame_ready = 0U;
        hdcmi.State = HAL_DCMI_STATE_READY;
        if (HAL_DCMI_Start_DMA(&hdcmi,
                               DCMI_MODE_SNAPSHOT,
                               (uint32_t)camera_dma_buf,
                               (FRAME_WIDTH * FRAME_HEIGHT * 2U) / 4U) != HAL_OK) {
          HAL_DCMI_Stop(&hdcmi);
          if (hdcmi.DMA_Handle != NULL) {
            HAL_DMA_Abort(hdcmi.DMA_Handle);
          }
          hdcmi.State = HAL_DCMI_STATE_READY;
          (void)HAL_DCMI_Start_DMA(&hdcmi,
                                   DCMI_MODE_SNAPSHOT,
                                   (uint32_t)camera_dma_buf,
                                   (FRAME_WIDTH * FRAME_HEIGHT * 2U) / 4U);
        }
        camera_dma_enabled = 1U;
      }
#endif
#if TEST_FULL_PIPELINE_ONE_SHOT
#if ENABLE_TENSOR_EQUIV_DUMP
      dump_tensor_equivalence_frame();
#endif
#if ENABLE_LCD_RGB565_DUMP
      dump_lcd_rgb565_frame(lcd_front_buf);
#endif
      printf("one_shot_done,m=%lu,i=%lu/%lu,o=%lu/%lu,lcd=%lu,io=%lu/%lu\r\n",
          display_mode,
          model_input_min,
          model_input_max,
          model_output_min,
          model_output_max,
          camera_display_ms,
          ai_input_addr,
          ai_output_addr);
      while (1) {
        HAL_Delay(1000);
      }
#endif
    }
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 60;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 20;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  boot_stage = 11;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  boot_stage = 12;

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;

  boot_stage = 13;
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
  boot_stage = 14;

  /** Enables the Clock Security System
  */
  HAL_RCC_EnableCSS();
}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{
  __HAL_RCC_DMA1_CLK_ENABLE();

  HAL_NVIC_SetPriority(DMA1_Stream0_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream0_IRQn);
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

 /* MPU Configuration */

void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct = {0};

  /* Disables the MPU */
  HAL_MPU_Disable();

  /** Initializes and configures the Region and the memory to be protected
  */
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER0;
  MPU_InitStruct.BaseAddress = 0x0;
  MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
  MPU_InitStruct.SubRegionDisable = 0x87;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;

  HAL_MPU_ConfigRegion(&MPU_InitStruct);
  /* Enables the MPU */
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);

}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  error_stage = boot_stage;
  if (boot_stage < 900U) {
    boot_stage += 9000U;
  }
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

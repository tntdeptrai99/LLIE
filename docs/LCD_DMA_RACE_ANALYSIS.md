# LCD DMA Race Analysis

Target firmware: `stm32/firmware/LLIE_E2E_Benchmark`

## Source Finding

The active ST7735 path does not currently use SPI DMA for the RGB image transfer.

Relevant source evidence:

- `Drivers/BSP/ST7735/st7735.c` implements `ST7735_FillRGBRect`.
- `ST7735_FillRGBRect` copies one row into an internal static byte buffer, then sends that row.
- `Drivers/BSP/ST7735/lcd.c` sends data through `HAL_SPI_Transmit`.
- No active call site was found for `HAL_SPI_Transmit_DMA` in the application LCD path.

Therefore the prompt's top two suspected causes:

```text
1. LCD DMA reads while CPU writes same frame buffer
2. Missing DCache clean before SPI DMA
```

are not supported by the current source.

## What This Means

If the board still shows LCD noise while camera-input mode and LCD test patterns are clean, the LCD transport itself is probably not the first bad point.

The more likely first bad point is one of:

- AI public output layout interpretation.
- AI output semantic interpretation: gain/residual scale, zero point, and blend math.
- Postprocess indexing.
- Full-pipeline camera race, if fixed input passes but camera pipeline fails.

## Race Still To Recheck If DMA Is Reintroduced

If SPI DMA is added later, the minimum correctness rules are:

```text
CPU writes LCD buffer -> clean DCache range -> SPI DMA reads buffer
Do not write that buffer again until DMA complete callback fires
```

For correctness-first testing, block until transfer complete or use a stateful double buffer. Any variable shared between main loop and DMA callbacks must be `volatile`.

## Current LCD-Specific Tests

The existing clean camera-input display result implies:

- ST7735 address window is mostly correct.
- RGB565 transport is not completely broken.
- Byte order is likely usable for camera path.

Remaining LCD-only checks are color order and exact RGB565 byte order, but those should be verified with red/green/blue/checkerboard test patterns before changing code.


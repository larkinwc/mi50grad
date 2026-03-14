/*
 * Fast fused allreduce+residual for TP inference.
 *
 * All HIP API calls + FP16 vector add in one C call.
 * Uses F16C + AVX for hardware FP16↔FP32 conversion.
 * Static buffers (BSS) — fastest for small 10KB payloads.
 *
 * Compile: gcc -O3 -mf16c -mavx -shared -fPIC
 */

#include <stdint.h>
#include <string.h>
#include <immintrin.h>

typedef int (*hipSetDevice_t)(int);
typedef int (*hipDeviceSynchronize_t)(void);
typedef int (*hipMemcpy_t)(void*, const void*, size_t, int);

static hipSetDevice_t p_hipSetDevice;
static hipDeviceSynchronize_t p_hipDeviceSynchronize;
static hipMemcpy_t p_hipMemcpy;

/* Static staging buffers — BSS, cache-warm, no allocation overhead */
#define MAX_HIDDEN 8192
static uint16_t g_partial0[MAX_HIDDEN] __attribute__((aligned(32)));
static uint16_t g_partial1[MAX_HIDDEN] __attribute__((aligned(32)));
static uint16_t g_result[MAX_HIDDEN] __attribute__((aligned(32)));

int fast_ar_init(void* set_device, void* sync, void* memcpy_fn) {
    p_hipSetDevice = (hipSetDevice_t)set_device;
    p_hipDeviceSynchronize = (hipDeviceSynchronize_t)sync;
    p_hipMemcpy = (hipMemcpy_t)memcpy_fn;
    return 0;
}

/*
 * Fused allreduce + residual for TP=2.
 *
 * One C call:
 * 1. D2H partial0 + hidden from GPU0
 * 2. D2H partial1 from GPU1
 * 3. AVX FP16 add: result = hidden + partial0 + partial1
 * 4. H2D result → GPU0, GPU1
 *
 * hipMemcpy(D2H) is synchronous — waits for prior GPU work.
 */
int fast_ar_fused_tp2(
    int dev0, int dev1,
    uint64_t d_partial0, uint64_t d_partial1,
    uint64_t d_hidden0, uint64_t d_hidden1,
    int num_elems)
{
    int err;
    int size = num_elems * 2;

    if ((err = p_hipSetDevice(dev0))) return err;
    if ((err = p_hipMemcpy(g_partial0, (void*)d_partial0, size, 2))) return err;
    if ((err = p_hipMemcpy(g_result, (void*)d_hidden0, size, 2))) return err;

    if ((err = p_hipSetDevice(dev1))) return err;
    if ((err = p_hipMemcpy(g_partial1, (void*)d_partial1, size, 2))) return err;

    /* AVX FP16 add: result = hidden + partial0 + partial1 */
    {
        int i = 0;
        for (; i + 7 < num_elems; i += 8) {
            __m128i hh = _mm_loadu_si128((const __m128i*)(g_result + i));
            __m128i hp0 = _mm_loadu_si128((const __m128i*)(g_partial0 + i));
            __m128i hp1 = _mm_loadu_si128((const __m128i*)(g_partial1 + i));
            __m256 fh = _mm256_cvtph_ps(hh);
            __m256 fp0 = _mm256_cvtph_ps(hp0);
            __m256 fp1 = _mm256_cvtph_ps(hp1);
            __m256 sum = _mm256_add_ps(_mm256_add_ps(fh, fp0), fp1);
            _mm_storeu_si128((__m128i*)(g_result + i),
                             _mm256_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT));
        }
        for (; i < num_elems; i++) {
            __m128i hh = _mm_set1_epi16(g_result[i]);
            __m128i hp0 = _mm_set1_epi16(g_partial0[i]);
            __m128i hp1 = _mm_set1_epi16(g_partial1[i]);
            __m128 fh = _mm_cvtph_ps(hh);
            __m128 fp0 = _mm_cvtph_ps(hp0);
            __m128 fp1 = _mm_cvtph_ps(hp1);
            __m128 sum = _mm_add_ps(_mm_add_ps(fh, fp0), fp1);
            __m128i hr = _mm_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT);
            g_result[i] = (uint16_t)_mm_extract_epi16(hr, 0);
        }
    }

    if ((err = p_hipSetDevice(dev0))) return err;
    if ((err = p_hipMemcpy((void*)d_hidden0, g_result, size, 1))) return err;
    if ((err = p_hipSetDevice(dev1))) return err;
    if ((err = p_hipMemcpy((void*)d_hidden1, g_result, size, 1))) return err;

    return 0;
}

/*
 * Allreduce partials only (no hidden download/add) for TP=2.
 * Writes sum of partials to d_dst0 and d_dst1.
 *
 * 1. Sync GPU0, D2H partial0
 * 2. Sync GPU1, D2H partial1
 * 3. AVX add: sum = partial0 + partial1
 * 4. H2D sum → d_dst0, d_dst1
 *
 * Caller then uses skip_rmsnorm to fuse residual add + normalize.
 * Saves 1 D2H (no hidden download) per call.
 */
int fast_ar_sum_tp2(
    int dev0, int dev1,
    uint64_t d_partial0, uint64_t d_partial1,
    uint64_t d_dst0, uint64_t d_dst1,
    int num_elems)
{
    int err;
    int size = num_elems * 2;

    /* hipMemcpy(D2H) is synchronous and waits for prior GPU work on default stream */
    if ((err = p_hipSetDevice(dev0))) return err;
    if ((err = p_hipMemcpy(g_partial0, (void*)d_partial0, size, 2))) return err;

    if ((err = p_hipSetDevice(dev1))) return err;
    if ((err = p_hipMemcpy(g_partial1, (void*)d_partial1, size, 2))) return err;

    /* AVX add: result = partial0 + partial1 */
    {
        int i = 0;
        for (; i + 7 < num_elems; i += 8) {
            __m128i hp0 = _mm_loadu_si128((const __m128i*)(g_partial0 + i));
            __m128i hp1 = _mm_loadu_si128((const __m128i*)(g_partial1 + i));
            __m256 fp0 = _mm256_cvtph_ps(hp0);
            __m256 fp1 = _mm256_cvtph_ps(hp1);
            __m256 sum = _mm256_add_ps(fp0, fp1);
            _mm_storeu_si128((__m128i*)(g_result + i),
                             _mm256_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT));
        }
        for (; i < num_elems; i++) {
            __m128i hp0 = _mm_set1_epi16(g_partial0[i]);
            __m128i hp1 = _mm_set1_epi16(g_partial1[i]);
            __m128 fp0 = _mm_cvtph_ps(hp0);
            __m128 fp1 = _mm_cvtph_ps(hp1);
            __m128 sum = _mm_add_ps(fp0, fp1);
            __m128i hr = _mm_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT);
            g_result[i] = (uint16_t)_mm_extract_epi16(hr, 0);
        }
    }

    if ((err = p_hipSetDevice(dev0))) return err;
    if ((err = p_hipMemcpy((void*)d_dst0, g_result, size, 1))) return err;
    if ((err = p_hipSetDevice(dev1))) return err;
    if ((err = p_hipMemcpy((void*)d_dst1, g_result, size, 1))) return err;

    return 0;
}

/*
 * Fused allreduce + residual for TP=3.
 * Downloads partials from 3 GPUs + hidden from GPU0,
 * AVX add: result = hidden + partial0 + partial1 + partial2,
 * uploads result to all 3 GPUs.
 */
static uint16_t g_partial2[MAX_HIDDEN] __attribute__((aligned(32)));

int fast_ar_fused_tp3(
    int dev0, int dev1, int dev2,
    uint64_t d_partial0, uint64_t d_partial1, uint64_t d_partial2,
    uint64_t d_hidden0, uint64_t d_hidden1, uint64_t d_hidden2,
    int num_elems)
{
    int err;
    int size = num_elems * 2;

    if ((err = p_hipSetDevice(dev0))) return err;
    if ((err = p_hipMemcpy(g_partial0, (void*)d_partial0, size, 2))) return err;
    if ((err = p_hipMemcpy(g_result, (void*)d_hidden0, size, 2))) return err;

    if ((err = p_hipSetDevice(dev1))) return err;
    if ((err = p_hipMemcpy(g_partial1, (void*)d_partial1, size, 2))) return err;

    if ((err = p_hipSetDevice(dev2))) return err;
    if ((err = p_hipMemcpy(g_partial2, (void*)d_partial2, size, 2))) return err;

    /* AVX FP16 add: result = hidden + partial0 + partial1 + partial2 */
    {
        int i = 0;
        for (; i + 7 < num_elems; i += 8) {
            __m128i hh = _mm_loadu_si128((const __m128i*)(g_result + i));
            __m128i hp0 = _mm_loadu_si128((const __m128i*)(g_partial0 + i));
            __m128i hp1 = _mm_loadu_si128((const __m128i*)(g_partial1 + i));
            __m128i hp2 = _mm_loadu_si128((const __m128i*)(g_partial2 + i));
            __m256 fh = _mm256_cvtph_ps(hh);
            __m256 fp0 = _mm256_cvtph_ps(hp0);
            __m256 fp1 = _mm256_cvtph_ps(hp1);
            __m256 fp2 = _mm256_cvtph_ps(hp2);
            __m256 sum = _mm256_add_ps(_mm256_add_ps(fh, fp0),
                                       _mm256_add_ps(fp1, fp2));
            _mm_storeu_si128((__m128i*)(g_result + i),
                             _mm256_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT));
        }
        for (; i < num_elems; i++) {
            __m128i hh = _mm_set1_epi16(g_result[i]);
            __m128i hp0 = _mm_set1_epi16(g_partial0[i]);
            __m128i hp1 = _mm_set1_epi16(g_partial1[i]);
            __m128i hp2 = _mm_set1_epi16(g_partial2[i]);
            __m128 fh = _mm_cvtph_ps(hh);
            __m128 fp0 = _mm_cvtph_ps(hp0);
            __m128 fp1 = _mm_cvtph_ps(hp1);
            __m128 fp2 = _mm_cvtph_ps(hp2);
            __m128 sum = _mm_add_ps(_mm_add_ps(fh, fp0),
                                    _mm_add_ps(fp1, fp2));
            __m128i hr = _mm_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT);
            g_result[i] = (uint16_t)_mm_extract_epi16(hr, 0);
        }
    }

    if ((err = p_hipSetDevice(dev0))) return err;
    if ((err = p_hipMemcpy((void*)d_hidden0, g_result, size, 1))) return err;
    if ((err = p_hipSetDevice(dev1))) return err;
    if ((err = p_hipMemcpy((void*)d_hidden1, g_result, size, 1))) return err;
    if ((err = p_hipSetDevice(dev2))) return err;
    if ((err = p_hipMemcpy((void*)d_hidden2, g_result, size, 1))) return err;

    return 0;
}

/*
 * Fused allreduce + residual for TP=4.
 * Downloads partials from 4 GPUs + hidden from GPU0,
 * AVX add: result = hidden + partial0 + partial1 + partial2 + partial3,
 * uploads result to all 4 GPUs.
 */
static uint16_t g_partial3[MAX_HIDDEN] __attribute__((aligned(32)));

int fast_ar_fused_tp4(
    int dev0, int dev1, int dev2, int dev3,
    uint64_t d_partial0, uint64_t d_partial1,
    uint64_t d_partial2, uint64_t d_partial3,
    uint64_t d_hidden0, uint64_t d_hidden1,
    uint64_t d_hidden2, uint64_t d_hidden3,
    int num_elems)
{
    int err;
    int size = num_elems * 2;

    if ((err = p_hipSetDevice(dev0))) return err;
    if ((err = p_hipMemcpy(g_partial0, (void*)d_partial0, size, 2))) return err;
    if ((err = p_hipMemcpy(g_result, (void*)d_hidden0, size, 2))) return err;

    if ((err = p_hipSetDevice(dev1))) return err;
    if ((err = p_hipMemcpy(g_partial1, (void*)d_partial1, size, 2))) return err;

    if ((err = p_hipSetDevice(dev2))) return err;
    if ((err = p_hipMemcpy(g_partial2, (void*)d_partial2, size, 2))) return err;

    if ((err = p_hipSetDevice(dev3))) return err;
    if ((err = p_hipMemcpy(g_partial3, (void*)d_partial3, size, 2))) return err;

    /* AVX FP16 add: result = hidden + partial0 + partial1 + partial2 + partial3 */
    {
        int i = 0;
        for (; i + 7 < num_elems; i += 8) {
            __m128i hh = _mm_loadu_si128((const __m128i*)(g_result + i));
            __m128i hp0 = _mm_loadu_si128((const __m128i*)(g_partial0 + i));
            __m128i hp1 = _mm_loadu_si128((const __m128i*)(g_partial1 + i));
            __m128i hp2 = _mm_loadu_si128((const __m128i*)(g_partial2 + i));
            __m128i hp3 = _mm_loadu_si128((const __m128i*)(g_partial3 + i));
            __m256 fh = _mm256_cvtph_ps(hh);
            __m256 fp0 = _mm256_cvtph_ps(hp0);
            __m256 fp1 = _mm256_cvtph_ps(hp1);
            __m256 fp2 = _mm256_cvtph_ps(hp2);
            __m256 fp3 = _mm256_cvtph_ps(hp3);
            __m256 sum = _mm256_add_ps(
                _mm256_add_ps(fh, fp0),
                _mm256_add_ps(_mm256_add_ps(fp1, fp2), fp3));
            _mm_storeu_si128((__m128i*)(g_result + i),
                             _mm256_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT));
        }
        for (; i < num_elems; i++) {
            __m128i hh = _mm_set1_epi16(g_result[i]);
            __m128i hp0 = _mm_set1_epi16(g_partial0[i]);
            __m128i hp1 = _mm_set1_epi16(g_partial1[i]);
            __m128i hp2 = _mm_set1_epi16(g_partial2[i]);
            __m128i hp3 = _mm_set1_epi16(g_partial3[i]);
            __m128 fh = _mm_cvtph_ps(hh);
            __m128 fp0 = _mm_cvtph_ps(hp0);
            __m128 fp1 = _mm_cvtph_ps(hp1);
            __m128 fp2 = _mm_cvtph_ps(hp2);
            __m128 fp3 = _mm_cvtph_ps(hp3);
            __m128 sum = _mm_add_ps(
                _mm_add_ps(fh, fp0),
                _mm_add_ps(_mm_add_ps(fp1, fp2), fp3));
            __m128i hr = _mm_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT);
            g_result[i] = (uint16_t)_mm_extract_epi16(hr, 0);
        }
    }

    if ((err = p_hipSetDevice(dev0))) return err;
    if ((err = p_hipMemcpy((void*)d_hidden0, g_result, size, 1))) return err;
    if ((err = p_hipSetDevice(dev1))) return err;
    if ((err = p_hipMemcpy((void*)d_hidden1, g_result, size, 1))) return err;
    if ((err = p_hipSetDevice(dev2))) return err;
    if ((err = p_hipMemcpy((void*)d_hidden2, g_result, size, 1))) return err;
    if ((err = p_hipSetDevice(dev3))) return err;
    if ((err = p_hipMemcpy((void*)d_hidden3, g_result, size, 1))) return err;

    return 0;
}

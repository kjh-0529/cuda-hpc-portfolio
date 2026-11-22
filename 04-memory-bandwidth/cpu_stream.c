/* CPU memory bandwidth: STREAM-style Copy, Scale, Add and Triad on doubles with OpenMP,
 * written from scratch in the style of John McCalpin's STREAM.
 *
 * Each array should be at least 4x the last-level cache, so the kernels measure
 * memory, not cache. Default: 2^27 doubles = 1.07 GB per array (3.2 GB in total).
 * Arrays are initialized by the same threads, with the same schedule, that later use
 * them, so pages are placed on the right NUMA node ("first touch").
 *
 * Bytes counted per element, as in STREAM: Copy/Scale 16, Add/Triad 24. Without
 * non-temporal stores the CPU also reads each destination line before writing it
 * (write-allocate), so real traffic is higher than the counted bytes.
 *
 * Usage: ./cpu_stream [ELEMENTS] [LABEL]   (threads: OMP_NUM_THREADS, placement: OMP_PROC_BIND)
 * Prints one CSV line per kernel; LABEL names the build, e.g. the compiler flags.
 */
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NTIMES 20
static const double kScalar = 3.0;

static int cmp(const void* x, const void* y) {
    double a = *(const double*)x, b = *(const double*)y;
    return (a > b) - (a < b);
}

int main(int argc, char** argv) {
    const size_t n = argc > 1 ? strtoull(argv[1], NULL, 10) : ((size_t)1 << 27);
    const char* label = argc > 2 ? argv[2] : "unlabeled";
    const char* bind = getenv("OMP_PROC_BIND") ? getenv("OMP_PROC_BIND") : "unset";
    double* a = malloc(n * sizeof(double));
    double* b = malloc(n * sizeof(double));
    double* c = malloc(n * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "out of memory\n"); return 1; }

#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < n; j++) { a[j] = 1.0; b[j] = 2.0; c[j] = 0.0; }

    const char* names[4] = {"copy", "scale", "add", "triad"};
    const double bytes[4] = {16.0 * n, 16.0 * n, 24.0 * n, 24.0 * n};
    double t[4][NTIMES];

    for (int k = 0; k < NTIMES; k++) {
        t[0][k] = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (size_t j = 0; j < n; j++) c[j] = a[j];
        t[0][k] = omp_get_wtime() - t[0][k];

        t[1][k] = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (size_t j = 0; j < n; j++) b[j] = kScalar * c[j];
        t[1][k] = omp_get_wtime() - t[1][k];

        t[2][k] = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (size_t j = 0; j < n; j++) c[j] = a[j] + b[j];
        t[2][k] = omp_get_wtime() - t[2][k];

        t[3][k] = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (size_t j = 0; j < n; j++) a[j] = b[j] + kScalar * c[j];
        t[3][k] = omp_get_wtime() - t[3][k];
    }

    /* Verify: replay the same sequence on scalars (relative tolerance for FMA rounding). */
    double ea = 1.0, eb = 2.0, ec = 0.0;
    for (int k = 0; k < NTIMES; k++) { ec = ea; eb = kScalar * ec; ec = ea + eb; ea = eb + kScalar * ec; }
    int ok = 1;
    for (size_t j = 0; j < n; j += n / 1000 + 1)
        ok &= fabs(a[j] - ea) <= 1e-10 * fabs(ea) && fabs(b[j] - eb) <= 1e-10 * fabs(eb) &&
              fabs(c[j] - ec) <= 1e-10 * fabs(ec);
    if (!ok) { fprintf(stderr, "verification FAILED (%s)\n", label); return 1; }

    /* As in STREAM, skip the first iteration; report the best and the median time. */
    for (int i = 0; i < 4; i++) {
        double s[NTIMES - 1];
        memcpy(s, &t[i][1], sizeof s);
        qsort(s, NTIMES - 1, sizeof(double), cmp);
        printf("%s,%d,%s,%s,%.1f,%.1f\n", label, omp_get_max_threads(), bind, names[i],
               bytes[i] / s[0] / 1e9, bytes[i] / s[(NTIMES - 1) / 2] / 1e9);
    }
    free(a); free(b); free(c);
    return 0;
}

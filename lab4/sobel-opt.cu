#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <zlib.h>
#include <png.h>

#define Z 2
#define Y 5
#define X 5
#define xBound (X / 2)
#define yBound (Y / 2)
#define SCALE 8

#define BLOCK_X 16
#define BLOCK_Y 16

int read_png(const char* filename, unsigned char** image, unsigned* height,
             unsigned* width, unsigned* channels) {
    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");
    if (!infile) return 1;

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8)) {
        fclose(infile);
        return 1;   /* bad signature */
    }

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(infile);
        return 4;   /* out of memory */
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(infile);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32 i, rowbytes;
    png_bytep row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(infile);
        return 3;
    }

    for (i = 0; i < *height; ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    fclose(infile);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width,
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__constant__ char mask[Z][Y][X] = { { { -1, -4, -6, -4, -1 },
                                      { -2, -8, -12, -8, -2 },
                                      { 0, 0, 0, 0, 0 },
                                      { 2, 8, 12, 8, 2 },
                                      { 1, 4, 6, 4, 1 } },
                                    { { -1, -2, 0, 2, 1 },
                                      { -4, -8, 0, 8, 4 },
                                      { -6, -12, 0, 12, 6 },
                                      { -4, -8, 0, 8, 4 },
                                      { -1, -2, 0, 2, 1 } } };

__global__ void sobel(const unsigned char *__restrict__ s, unsigned char *__restrict__ t, 
                      unsigned height, unsigned width, unsigned channels) {
    // Shared memory tile including halo region: (BLOCK_X + 4) x (BLOCK_Y + 4) with 3 channels
    __shared__ unsigned char shared_img[BLOCK_Y + 4][(BLOCK_X + 4) * 3];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK_X + tx;
    int y = blockIdx.y * BLOCK_Y + ty;

    int total_threads = BLOCK_X * BLOCK_Y;
    int thread_id = ty * BLOCK_X + tx;
    int tile_w = BLOCK_X + 4;
    int tile_h = BLOCK_Y + 4;
    int total_pixels = tile_w * tile_h;

    // Colevtively load global memory into shared memory with halo padding
    for (int i = thread_id; i < total_pixels; i += total_threads) {
        int smem_y = i / tile_w;
        int smem_x = i % tile_w;

        int global_x = blockIdx.x * BLOCK_X + smem_x - xBound;
        int global_y = blockIdx.y * BLOCK_Y + smem_y - yBound;

        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
            int global_idx = channels * (width * global_y + global_x);
            shared_img[smem_y][smem_x * 3 + 0] = s[global_idx + 0]; // B
            shared_img[smem_y][smem_x * 3 + 1] = s[global_idx + 1]; // G
            shared_img[smem_y][smem_x * 3 + 2] = s[global_idx + 2]; // R
        } else {
            shared_img[smem_y][smem_x * 3 + 0] = 0;
            shared_img[smem_y][smem_x * 3 + 1] = 0;
            shared_img[smem_y][smem_x * 3 + 2] = 0;
        }
    }

    __syncthreads();

    if (x < width && y < height) {
        float val[Z][3];
        for (int i = 0; i < Z; ++i) {
            val[i][0] = 0.f;
            val[i][1] = 0.f;
            val[i][2] = 0.f;

            for (int v = -yBound; v <= yBound; ++v) {
                for (int u = -xBound; u <= xBound; ++u) {
                    int smem_y = ty + yBound + v;
                    int smem_x = tx + xBound + u;

                    unsigned char B = shared_img[smem_y][smem_x * 3 + 0];
                    unsigned char G = shared_img[smem_y][smem_x * 3 + 1];
                    unsigned char R = shared_img[smem_y][smem_x * 3 + 2];

                    char m = mask[i][u + xBound][v + yBound];
                    val[i][0] += B * m;
                    val[i][1] += G * m;
                    val[i][2] += R * m;
                }
            }
        }

        float totalR = 0.f;
        float totalG = 0.f;
        float totalB = 0.f;

        for (int i = 0; i < Z; ++i) {
            totalR += val[i][2] * val[i][2];
            totalG += val[i][1] * val[i][1];
            totalB += val[i][0] * val[i][0];
        }

        totalR = sqrtf(totalR) / SCALE;
        totalG = sqrtf(totalG) / SCALE;
        totalB = sqrtf(totalB) / SCALE;

        const unsigned char cR = (totalR > 255.f) ? 255 : (unsigned char)totalR;
        const unsigned char cG = (totalG > 255.f) ? 255 : (unsigned char)totalG;
        const unsigned char cB = (totalB > 255.f) ? 255 : (unsigned char)totalB;

        int out_idx = channels * (width * y + x);
        t[out_idx + 2] = cR;
        t[out_idx + 1] = cG;
        t[out_idx + 0] = cB;
    }
}

int main(int argc, char **argv) {
    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char *src = NULL, *dst;
    unsigned char *dsrc, *ddst;

    /* read the image to src, and get height, width, channels */
    if (read_png(argv[1], &src, &height, &width, &channels)) {
        std::cerr << "Error in read png" << std::endl;
        return -1;
    }

    dst = (unsigned char *)malloc(height * width * channels * sizeof(unsigned char));
    cudaHostRegister(src, height * width * channels * sizeof(unsigned char), cudaHostRegisterDefault);

    // cudaMalloc(...) for device src and device dst
    cudaMalloc(&dsrc, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&ddst, height * width * channels * sizeof(unsigned char));

    // cudaMemcpy(...) copy source image to device
    cudaMemcpy(dsrc, src, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define 2D block and grid dimensions
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);

    // launch cuda kernel
    sobel<<<grid, block>>>(dsrc, ddst, height, width, channels);

    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(dst, ddst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    write_png(argv[2], dst, height, width, channels);
    free(src);
    free(dst);
    cudaFree(dsrc);
    cudaFree(ddst);
    return 0;
}
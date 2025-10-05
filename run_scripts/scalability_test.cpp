#include <liburing.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <chrono>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <random>
#include <climits>
#include <sys/mman.h>
#include <mpi.h>
#include <unistd.h> 
#include <cstddef> 
#include <fstream> 
// #include <cuda_runtime.h>
#include <json.hpp>
using json = nlohmann::json;


enum class FileMode {
    ChunkPerFile,   // (1)
    OnePerGPU,      // (2)
    TwoGPUsPerFile, // (3)
    AllGPUsOneFile  // (4)
};

constexpr size_t KB = 1024;
constexpr size_t MB = 1024 * KB;
constexpr int QUEUE_DEPTH = 512;   // io_uring queue depth
constexpr size_t CHUNK_SIZE = 128 * MB; // size of each write chunk

// Random buffer fill
void fill_random(void* buf, size_t size) {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<unsigned long long> dist(0, ULLONG_MAX);

    auto* p = static_cast<unsigned long long*>(buf);
    size_t words = size / sizeof(unsigned long long);
    for (size_t i = 0; i < words; i++) {
        p[i] = dist(rng);
    }

    unsigned char* tail = reinterpret_cast<unsigned char*>(p + words);
    for (size_t j = words * sizeof(unsigned long long); j < size; j++) {
        tail[j - words * sizeof(unsigned long long)] = dist(rng) & 0xFF;
    }
}

// Allocate pinned aligned buffer
void* alloc_pinned(size_t size, size_t alignment = 4096) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        perror("posix_memalign");
        exit(1);
    }
    // cudaHostRegister(ptr, size, cudaHostRegisterPortable);
    fill_random(ptr, size);

    if (mlock(ptr, size) != 0) {
        perror("mlock"); // not fatal, but memory not pinned
    }
    return ptr;
}

double run_io_uring_write(const char* path, void* buf, size_t size, int rank=0, size_t offset=0) {
    io_uring ring;
    if (io_uring_queue_init(QUEUE_DEPTH, &ring, 0) < 0) {
        perror("io_uring_queue_init");
        exit(1);
    }

    int fd = open(path, O_CREAT | O_RDWR | O_DIRECT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open");
        exit(1);
    }

    // cudaSetDevice(rank % 4); // assuming max 4 GPUs per node
    // cudaDeviceSynchronize();
    // char* cuda_buf;
    // cudaMalloc(&cuda_buf, size);
    // cudaMemset(cuda_buf, 0, size);
    // cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD); // sync before start
    double start = MPI_Wtime();

    // cudaMemcpy(cuda_buf, buf, size, cudaMemcpyHostToDevice);
    // int num_threads = 32;
    // size_t chunk_size = size / num_threads;
    // #pragma omp parallel for num_threads(num_threads)
    // for (int i = 0; i < QUEUE_DEPTH; i++) {
    //     size_t offset = i * chunk_size;
    //     pwrite(fd, (char*)buf + offset, chunk_size, offset); // fallback to simple write for now
    // }

    size_t submitted = 0, written = 0;
    size_t chunk = CHUNK_SIZE;

    while (written < size) {
        unsigned to_submit = 0;

        while (submitted < size && to_submit < QUEUE_DEPTH) {
            size_t this_chunk = std::min(chunk, size - submitted);

            io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            assert(sqe);

            io_uring_prep_write(sqe, fd,
                                (char*)buf + submitted,
                                this_chunk,
                                offset + submitted);

            submitted += this_chunk;
            to_submit++;
        }
        io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        io_uring_prep_fsync(sqe, fd, 0);
        io_uring_submit_and_wait(&ring, 1);
        to_submit++;

        int ret = io_uring_submit(&ring);
        if (ret < 0) {
            perror("io_uring_submit");
            exit(1);
        }

        for (unsigned i = 0; i < to_submit; i++) {
            io_uring_cqe* cqe;
            int ret = io_uring_wait_cqe(&ring, &cqe);
            if (ret < 0) {
                perror("io_uring_wait_cqe");
                exit(1);
            }
            if (cqe->res < 0) {
                std::cerr << "Async write failed: " << strerror(-cqe->res) << std::endl;
                exit(1);
            }
            written += cqe->res;
            io_uring_cqe_seen(&ring, cqe);
        }
    }

    fsync(fd);
    double end = MPI_Wtime();
    close(fd);
    io_uring_queue_exit(&ring);

    return end - start;
}


std::string strategy_name(FileMode mode) {
    switch (mode) {
        case FileMode::ChunkPerFile: return "chunk_per_file";
        case FileMode::OnePerGPU: return "one_per_gpu";
        case FileMode::TwoGPUsPerFile: return "two_gpus_per_file";
        case FileMode::AllGPUsOneFile: return "all_gpus_one_file";
        default: return "unknown";
    }
}

int main(int argc, char** argv) {
    std::string test_str = argv[1];

    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<FileMode> strategies = {
        FileMode::ChunkPerFile,
        FileMode::OnePerGPU,
        FileMode::TwoGPUsPerFile,
        FileMode::AllGPUsOneFile
    };

    std::vector<size_t> sizes = {128, 248, 512, 1024, 2048, 4096, 8192, 11264}; // MB (shorter for test)
    constexpr int trials = 2;
    std::string base_path = "/grand/VeloC/mikailg/file_scalability";

    if (rank==0) mkdir(base_path.c_str(), 0755);

    json results;

    for (auto mode : strategies) {
        std::string approach = strategy_name(mode);

        for (size_t mb : sizes) {
            size_t size = mb * MB;

            std::vector<double> local_times;
            for (int t = 0; t < trials; t++) {
                void* buf = alloc_pinned(size);
                MPI_Barrier(MPI_COMM_WORLD);

                std::string filename;
                size_t offset = 0;

                switch (mode) {
                    case FileMode::ChunkPerFile: {
                        // each chunk is its own file
                        size_t n_chunks = (size + CHUNK_SIZE - 1) / CHUNK_SIZE;
                        double total_time = 0.0;
                        for (size_t c = 0; c < n_chunks; c++) {
                            size_t chunk_size = std::min(CHUNK_SIZE, size - c*CHUNK_SIZE);
                            filename = base_path + "/" + approach + "_rank" + std::to_string(rank)
                                        + "_size" + std::to_string(mb) + "_trial" + std::to_string(t)
                                        + "_chunk" + std::to_string(c) + ".bin";
                            total_time += run_io_uring_write(filename.c_str(),
                                                             (char*)buf + c*CHUNK_SIZE,
                                                             chunk_size, 0);
                        }
                        local_times.push_back(total_time);
                        break;
                    }
                    case FileMode::OnePerGPU: {
                        filename = base_path + "/" + approach + "_rank" + std::to_string(rank)
                                   + "_size" + std::to_string(mb) + "_trial" + std::to_string(t) + ".bin";
                        double sec = run_io_uring_write(filename.c_str(), buf, size, 0);
                        local_times.push_back(sec);
                        break;
                    }
                    case FileMode::TwoGPUsPerFile: {
                        int group = rank / 2; // 2 ranks per file
                        filename = base_path + "/" + approach + "_group" + std::to_string(group)
                                   + "_size" + std::to_string(mb) + "_trial" + std::to_string(t) + ".bin";
                        offset = (rank % 2) * size; // rank 0 writes [0..size), rank1 [size..2*size)
                        double sec = run_io_uring_write(filename.c_str(), buf, size, offset);
                        local_times.push_back(sec);
                        break;
                    }
                    case FileMode::AllGPUsOneFile: {
                        filename = base_path + "/" + approach + "_all_size" + std::to_string(mb)
                                   + "_trial" + std::to_string(t) + ".bin";
                        offset = rank * size;
                        double sec = run_io_uring_write(filename.c_str(), buf, size, offset);
                        local_times.push_back(sec);
                        break;
                    }
                }

                free(buf);
            }

            std::vector<double> all_times(world_size * trials);
            MPI_Gather(local_times.data(), trials, MPI_DOUBLE,
                       all_times.data(), trials, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                json rank_results = json::array();
                for (int r = 0; r < world_size; r++) {
                    std::vector<double> rank_times(all_times.begin() + r*trials,
                                                   all_times.begin() + (r+1)*trials);
                    json one_rank = {
                        {std::to_string(mb), {
                            {approach, rank_times}
                        }}
                    };
                    rank_results.push_back(one_rank);
                }
                results[approach][std::to_string(mb)]["rank_results"] = rank_results;
            }
        }
    }

    if (rank == 0) {
        std::string out_file = test_str + "-results.json";
        std::ofstream ofs(out_file);
        if (!ofs) {
            std::cerr << "Error: could not open output file " << out_file << std::endl;
        } else {
            ofs << results.dump(2) << std::endl;
            ofs.close();
            std::cout << "Results written to " << out_file << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}

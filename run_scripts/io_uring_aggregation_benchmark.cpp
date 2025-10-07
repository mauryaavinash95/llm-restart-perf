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
constexpr int RING_PER_NODE = 1;
constexpr int RING_PER_PROC = 2;
constexpr int RING_PER_2NODES = 3;

struct chunk_writes_t {
    int fd; 
    size_t offset;
    size_t size;
    void *buf;
};

struct write_req {
    int fd;
    char* buf;
    size_t offset;
    size_t remaining;
};

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
    fill_random(ptr, size);
    if (mlock(ptr, size) != 0) {
        perror("mlock"); // not fatal, but memory not pinned
    }
    return ptr;
}

std::vector<chunk_writes_t> prep_writes(const std::string& base_path, int rank, int trial, size_t mb, size_t total_size, void* buf) {
    size_t n_chunks = (total_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
    std::vector<chunk_writes_t> chunks;
    for (size_t c = 0; c < n_chunks; c++) {
        size_t chunk_size = std::min(CHUNK_SIZE, total_size - c * CHUNK_SIZE);
        std::string filename = base_path + "/ChunkPerFile" + "_rank" +
                               std::to_string(rank) + "_size" + std::to_string(mb) +
                               "_trial" + std::to_string(trial) + "_chunk" +
                               std::to_string(c) + ".bin";
        int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_DIRECT | O_TRUNC, 0644);
        if (fd < 0) {
            perror("open");
            exit(1);
        }
        chunks.push_back({fd, 0, chunk_size, static_cast<char*>(buf) + c * CHUNK_SIZE});
    }
    return chunks;
}

void submit_writes(io_uring& ring, const std::vector<chunk_writes_t>& chunks) {
    for (auto& c : chunks) {
        write_req* req = new write_req{c.fd, static_cast<char*>(c.buf), c.offset, c.size};
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        if (!sqe) {
            std::cerr << "SQE ring full — increase QUEUE_DEPTH\n";
            break;
        }
        io_uring_prep_write(sqe, c.fd, req->buf, req->remaining, req->offset);
        io_uring_sqe_set_data(sqe, req);
    }
    io_uring_submit(&ring);
}

// Wait for completions and handle partial writes
void wait_for_complete(io_uring& ring, unsigned submitted_req) {
    struct io_uring_cqe* cqe;
    unsigned processed = 0;
    while (processed < submitted_req) {
        int ret = io_uring_wait_cqe(&ring, &cqe);
        if (ret < 0) {
            perror("io_uring_wait_cqe");
            exit(1);
        }

        write_req* req = static_cast<write_req*>(io_uring_cqe_get_data(cqe));
        ssize_t written = cqe->res;
        io_uring_cqe_seen(&ring, cqe);
        processed++;

        if (written < 0) {
            std::cerr << "Write failed: " << strerror(-written) << "\n";
            delete req;
            exit(1);
        }

        req->buf += written;
        req->offset += written;
        req->remaining -= written;

        if (req->remaining > 0) {
            // partial write, resubmit
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            assert(sqe);
            io_uring_prep_write(sqe, req->fd, req->buf, req->remaining, req->offset);
            io_uring_sqe_set_data(sqe, req);
            io_uring_submit(&ring);
            submitted_req++;
        } else {
            // fully written
            delete req;
        }
    }
}


double run_io_uring_write(io_uring& ring, int fd, void* buf, size_t size, size_t offset=0) {
    size_t submitted = 0;
    unsigned to_submit = 0;
    size_t chunk = CHUNK_SIZE;
    double start = MPI_Wtime();
    while (submitted < size && to_submit < QUEUE_DEPTH) {
        size_t this_chunk = std::min(chunk, size - submitted);
        write_req* req = new write_req{fd, static_cast<char*>(buf) + submitted, offset + submitted, this_chunk};
        io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        assert(sqe);
        io_uring_prep_write(sqe, fd, req->buf, req->remaining, req->offset);
        io_uring_sqe_set_data(sqe, req);
        submitted += this_chunk;
        to_submit++;
    }

    int ret = io_uring_submit(&ring);
    if (ret < 0) {
        perror("io_uring_submit");
        exit(1);
    }

    wait_for_complete(ring, to_submit);

    double end = MPI_Wtime();
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
    MPI_Init(&argc, &argv);
    std::string test_str = argv[1];

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<FileMode> strategies = {
        FileMode::ChunkPerFile,
        FileMode::OnePerGPU,
        FileMode::TwoGPUsPerFile,
        FileMode::AllGPUsOneFile
    };

    std::vector<size_t> sizes = {128, 256, 512, 1024}; // MB (shorter for test)
    constexpr int trials = 2;
    std::string base_path = "/grand/VeloC/mikailg/file_scalability";

    if (rank==0) 
        mkdir(base_path.c_str(), 0755);

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
                int fd = -1;

                switch (mode) {
                    case FileMode::ChunkPerFile: {
                        // each chunk is its own file
                        io_uring ring; 
                        if (io_uring_queue_init(QUEUE_DEPTH, &ring, 0) < 0) { 
                            perror("io_uring_queue_init"); 
                            exit(1); 
                        }
                        auto chunks = prep_writes(base_path, rank, t, mb, size, buf);
                        double start = MPI_Wtime();
                        submit_writes(ring, chunks);
                        wait_for_complete(ring, chunks.size());
                        for(auto &c : chunks){
                            fsync(c.fd);
                            close(c.fd);
                        }
                        double end = MPI_Wtime();
                        local_times.push_back(end-start);
                        io_uring_queue_exit(&ring);
                        break;
                    }
                    case FileMode::OnePerGPU: {
                        io_uring ring; 
                        if (io_uring_queue_init(QUEUE_DEPTH, &ring, 0) < 0) { 
                            perror("io_uring_queue_init"); 
                            exit(1); 
                        }
                        filename = base_path + "/" + approach + "_rank" + std::to_string(rank)
                                   + "_size" + std::to_string(mb) + "_trial" + std::to_string(t) + ".bin";
                        fd = open(filename.c_str(), O_CREAT | O_RDWR | O_DIRECT | O_TRUNC, 0644); 
                        if (fd < 0) { 
                            perror("open"); 
                            exit(1); 
                        }
                        double sec = run_io_uring_write(ring, fd, buf, size, 0);
                        double start = MPI_Wtime();
                        fsync(fd);
                        double sync_time = MPI_Wtime() - start;
                        close(fd);
                        io_uring_queue_exit(&ring);
                        local_times.push_back(sec + sync_time);
                        break;
                    }
                    case FileMode::TwoGPUsPerFile: {
                        io_uring ring; 
                        if (io_uring_queue_init(QUEUE_DEPTH, &ring, 0) < 0) { 
                            perror("io_uring_queue_init"); 
                            exit(1); 
                        }
                        int group = rank / 2; // 2 ranks per file
                        filename = base_path + "/" + approach + "_group" + std::to_string(group)
                                   + "_size" + std::to_string(mb) + "_trial" + std::to_string(t) + ".bin";
                        offset = (rank % 2) * size; // rank 0 writes [0..size), rank1 [size..2*size)
                        if((rank % 2) == 0) 
                            fd = open(filename.c_str(), O_CREAT | O_RDWR | O_DIRECT | O_TRUNC, 0644); 
                        else
                            fd = open(filename.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0644); 
                        if (fd < 0) { 
                            perror("open"); 
                            exit(1); 
                        }
                        MPI_Barrier(MPI_COMM_WORLD);
                        double sec = run_io_uring_write(ring, fd, buf, size, offset);
                        double start = MPI_Wtime();
                        fsync(fd);
                        double sync_time = MPI_Wtime() - start;
                        close(fd);
                        io_uring_queue_exit(&ring);
                        local_times.push_back(sec + sync_time);
                        break;
                    }
                    case FileMode::AllGPUsOneFile: {
                        io_uring ring; 
                        if (io_uring_queue_init(QUEUE_DEPTH, &ring, 0) < 0) { 
                            perror("io_uring_queue_init"); 
                            exit(1); 
                        }
                        filename = base_path + "/" + approach + "_all_size" + std::to_string(mb)
                                   + "_trial" + std::to_string(t) + ".bin";
                        offset = rank * size;
                        if(rank == 0) {
                            fd = open(filename.c_str(), O_CREAT | O_RDWR | O_DIRECT | O_TRUNC, 0644); 
                        }
                        else{
                            fd = open(filename.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0644); 
                        }
                        if (fd < 0) { 
                            perror("open"); 
                            exit(1); 
                        }
                        MPI_Barrier(MPI_COMM_WORLD);
                        double sec = run_io_uring_write(ring, fd, buf, size, offset);
                        double start = MPI_Wtime();
                        fsync(fd);
                        double sync_time = MPI_Wtime() - start;
                        close(fd);
                        io_uring_queue_exit(&ring);
                        local_times.push_back(sec + sync_time);
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

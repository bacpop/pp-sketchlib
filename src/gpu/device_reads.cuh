
class DeviceReads {
public:
  DeviceReads(const SeqBuf &seq_in, const size_t n_threads)
  : seq(std::make_unique<SeqBuf>(seq_in)),
      n_reads(seq_in.n_full_seqs()), read_length(seq_in.max_length()),
      current_block(0), buffer_filled(0), loaded_first(false) {
    // Set up buffer to load in reads (on host)
    size_t mem_free = 0;
    size_t mem_total = 0;
    CUDA_CALL(cudaMemGetInfo(&mem_free, &mem_total));
    buffer_size = (mem_free * 0.9) / (read_length * sizeof(char));
    buffer_blocks = std::floor(n_reads / (static_cast<double>(buffer_size) + 1)) + 1;
    if (buffer_size > n_reads) {
      buffer_size = n_reads;
      buffer_blocks = 1;
    }
    host_buffer.resize(buffer_size * read_length);
    CUDA_CALL(cudaHostRegister(
                host_buffer.data(),
                host_buffer.size() * sizeof(char),
                cudaHostRegisterDefault));

    // Buffer to store reads (on device)
    CUDA_CALL(cudaMalloc((void **)&d_reads,
                          buffer_size * read_length * sizeof(char)));

    CUDA_CALL(cudaStreamCreate(&memory_stream));
  }

  ~DeviceReads() {
    CUDA_CALL(cudaHostUnregister(host_buffer.data()));
    CUDA_CALL(cudaFree(d_reads));
    CUDA_CALL(cudaStreamDestroy(memory_stream));
  }

  bool next_buffer() {
    bool success;
    if (current_block < buffer_blocks) {
      if (buffer_blocks > 1 || !loaded_first) {
        size_t start = current_block * buffer_size;
        size_t end = (current_block + 1) * buffer_size;
        if (end > seq->n_full_seqs()) {
          end = seq->n_full_seqs();
        }
        buffer_filled = end - start;

        seq->load_seqs(host_buffer, start, end);
        CUDA_CALL(cudaMemcpyAsync(d_reads,
                                  host_buffer.data(),
                                  buffer_filled * read_length * sizeof(char),
                                  cudaMemcpyDefault,
                                  memory_stream));
        loaded_first = true;
      }
      current_block++;
      success = true;
    } else {
      success = false;
    }
    return success;
  }

  void reset_buffer() { current_block = 0; }

  char *read_ptr() { return d_reads; }
  size_t buffer_count() const { return buffer_filled; }
  size_t length() const { return read_length; }

  cudaStream_t stream() { return memory_stream; }

private:
  // delete move and copy to avoid accidentally using them
  DeviceReads(const DeviceReads &) = delete;
  DeviceReads(DeviceReads &&) = delete;

  char *d_reads;
  std::vector<char> host_buffer;
  std::unique_ptr<SeqBuf> seq;

  size_t n_reads;
  size_t read_length;
  size_t buffer_size;
  size_t buffer_blocks;
  size_t current_block;
  size_t buffer_filled;
  bool loaded_first;

  cudaStream_t memory_stream;
};

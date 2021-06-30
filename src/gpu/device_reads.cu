
#include "device_reads.cuh"
#include "cuda.cuh"

DeviceReads::DeviceReads(const SeqBuf &seq_in, const size_t n_threads)
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
}

DeviceReads::~DeviceReads() {
  CUDA_CALL(cudaHostUnregister(host_buffer.data()));
  CUDA_CALL(cudaFree(d_reads));
}

bool DeviceReads::next_buffer() {
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
                                cudaMemcpyDefault));
      loaded_first = true;
    }
    current_block++;
    success = true;
  } else {
    success = false;
  }
  return success;
}

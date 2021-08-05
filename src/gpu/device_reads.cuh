#pragma once

#include <memory> // unique_ptr
#include <vector>

#include "sketch/seqio.hpp" // SeqBuf

class DeviceReads {
public:
  DeviceReads(const std::shared_ptr<SeqBuf> &seq_in, const size_t n_threads);
  ~DeviceReads();

  bool next_buffer();

  void reset_buffer() { current_block = 0; }
  char *read_ptr() { return d_reads; }
  size_t buffer_count() const { return buffer_filled; }
  size_t length() const { return read_length; }

private:
  // delete copy and move to avoid accidentally using them
  DeviceReads(const DeviceReads &) = delete;
  DeviceReads(DeviceReads &&other) = delete;

  char *d_reads;
  std::vector<char> host_buffer;
  std::shared_ptr<SeqBuf> seq;

  size_t n_reads;
  size_t read_length;
  size_t buffer_size;
  size_t buffer_blocks;
  size_t current_block;
  size_t buffer_filled;
  bool loaded_first;
};

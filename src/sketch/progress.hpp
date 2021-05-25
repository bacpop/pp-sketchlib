#pragma once

class ProgressMeter {
public:
  ProgressMeter(size_t total) : total_(total), count_(0) { tick(0); }

  void tick(size_t blocks) {
    count_ += blocks;
    double progress = count_ / static_cast<double>(total_);
    progress = progress > 1 ? 1 : progress;
    fprintf(stderr, "%cProgress (CPU): %.1lf%%", 13, progress * 100);
  }

  void finalise() { fprintf(stderr, "%cProgress (CPU): 100.0%%\n", 13); }

private:
  size_t total_;
  volatile size_t count_;
};
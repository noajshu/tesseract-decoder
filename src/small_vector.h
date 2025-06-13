// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TESSERACT_SMALL_VECTOR_H
#define TESSERACT_SMALL_VECTOR_H

#include <cstddef>
#include <algorithm>

// A simple small buffer optimized vector for trivially copyable types.
// For sizes up to InlineSize elements the storage is kept on the stack.
// When the vector grows beyond this threshold it falls back to dynamic
// allocation.
template <typename T, size_t InlineSize = 128>
class SmallVector {
 public:
  SmallVector() : data_(inline_storage_), size_(0), capacity_(InlineSize) {}

  explicit SmallVector(size_t count, const T& value = T()) : SmallVector() {
    resize(count, value);
  }

  SmallVector(const SmallVector& other) : SmallVector() { *this = other; }

  SmallVector& operator=(const SmallVector& other) {
    if (this == &other) return *this;
    clear();
    if (other.size_ > InlineSize) {
      ensure_capacity(other.size_);
    }
    for (size_t i = 0; i < other.size_; ++i) {
      data_[i] = other.data_[i];
    }
    size_ = other.size_;
    return *this;
  }

  SmallVector(SmallVector&& other) noexcept : data_(inline_storage_), size_(0), capacity_(InlineSize) {
    if (other.data_ == other.inline_storage_) {
      std::copy(other.inline_storage_, other.inline_storage_ + other.size_, inline_storage_);
      size_ = other.size_;
    } else {
      data_ = other.data_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      other.data_ = other.inline_storage_;
      other.size_ = 0;
      other.capacity_ = InlineSize;
    }
  }

  SmallVector& operator=(SmallVector&& other) noexcept {
    if (this != &other) {
      if (data_ != inline_storage_) {
        delete[] data_;
      }
      if (other.data_ == other.inline_storage_) {
        data_ = inline_storage_;
        capacity_ = InlineSize;
        std::copy(other.inline_storage_, other.inline_storage_ + other.size_, inline_storage_);
        size_ = other.size_;
      } else {
        data_ = other.data_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        other.data_ = other.inline_storage_;
        other.size_ = 0;
        other.capacity_ = InlineSize;
      }
    }
    return *this;
  }

  ~SmallVector() {
    if (data_ != inline_storage_) {
      delete[] data_;
    }
  }

  void push_back(const T& value) {
    if (size_ == capacity_) {
      ensure_capacity(capacity_ * 2);
    }
    data_[size_++] = value;
  }

  void clear() { size_ = 0; }

  void resize(size_t count, const T& value = T()) {
    if (count > capacity_) {
      ensure_capacity(count);
    }
    for (size_t i = size_; i < count; ++i) {
      data_[i] = value;
    }
    size_ = count;
  }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  T& operator[](size_t idx) { return data_[idx]; }
  const T& operator[](size_t idx) const { return data_[idx]; }

  T* data() { return data_; }
  const T* data() const { return data_; }

  T* begin() { return data_; }
  const T* begin() const { return data_; }
  T* end() { return data_ + size_; }
  const T* end() const { return data_ + size_; }

 private:
  void ensure_capacity(size_t new_cap) {
    if (new_cap <= capacity_) return;
    T* new_data = new T[new_cap];
    std::copy(data_, data_ + size_, new_data);
    if (data_ != inline_storage_) {
      delete[] data_;
    }
    data_ = new_data;
    capacity_ = new_cap;
  }

  T* data_;
  size_t size_;
  size_t capacity_;
  T inline_storage_[InlineSize];
};

#endif  // TESSERACT_SMALL_VECTOR_H

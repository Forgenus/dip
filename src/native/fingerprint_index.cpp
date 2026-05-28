#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {
constexpr std::array<char, 8> kMagic = {'F', 'P', 'D', 'B', 'I', 'N', '1', '\0'};
constexpr uint32_t kVersion = 1;

template <typename T>
void write_value(std::ofstream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) {
        throw std::runtime_error("Failed to write fingerprint binary file");
    }
}

template <typename T>
T read_value(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("Failed to read fingerprint binary file");
    }
    return value;
}
}

class FingerprintIndex {
public:
    void insert(uint64_t address, uint64_t hash_value) {
        db_[address].push_back(hash_value);
        ++total_entries_;
        changed_ = true;
    }

    void insert_many(uint64_t address, const std::vector<uint64_t>& hash_values) {
        auto& entries = db_[address];
        entries.insert(entries.end(), hash_values.begin(), hash_values.end());
        total_entries_ += hash_values.size();
        changed_ = true;
    }

    std::vector<uint64_t> lookup(uint64_t address) const {
        auto it = db_.find(address);

        if (it == db_.end()) {
            return {};
        }

        return it->second;
    }

    std::vector<std::pair<uint64_t, uint64_t>> lookup_flat_batch(
        const std::vector<uint64_t>& addresses
    ) const {
        std::vector<std::pair<uint64_t, uint64_t>> result;

        for (uint64_t address : addresses) {
            auto it = db_.find(address);

            if (it == db_.end()) {
                continue;
            }

            for (uint64_t hash_value : it->second) {
                result.emplace_back(address, hash_value);
            }
        }

        return result;
    }

    bool contains(uint64_t address) const {
        return db_.find(address) != db_.end();
    }

    void remove(uint64_t address) {
        auto it = db_.find(address);

        if (it != db_.end()) {
            total_entries_ -= it->second.size();
            db_.erase(it);
        }

        changed_ = true;
    }

    void clear() {
        db_.clear();
        total_entries_ = 0;
        changed_ = true;
    }

    uint64_t size() const {
        return total_entries_;
    }

    uint64_t unique_addresses() const {
        return db_.size();
    }

    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> items() const {
        std::vector<std::pair<uint64_t, std::vector<uint64_t>>> result;
        result.reserve(db_.size());

        for (const auto& item : db_) {
            result.emplace_back(item.first, item.second);
        }

        return result;
    }

    void load_items(const std::vector<std::pair<uint64_t, std::vector<uint64_t>>>& items) {
        db_.clear();
        total_entries_ = 0;

        for (const auto& item : items) {
            auto existing = db_.find(item.first);

            if (existing != db_.end()) {
                total_entries_ -= existing->second.size();
            }

            db_[item.first] = item.second;
            total_entries_ += item.second.size();
        }

        changed_ = false;
    }

    void save_binary(const std::string& filename) {
        std::ofstream out(filename, std::ios::binary);

        if (!out) {
            throw std::runtime_error("Failed to open fingerprint binary file for writing");
        }

        out.write(kMagic.data(), kMagic.size());
        if (!out) {
            throw std::runtime_error("Failed to write fingerprint binary file");
        }

        write_value<uint32_t>(out, kVersion);
        write_value<uint64_t>(out, db_.size());
        write_value<uint64_t>(out, total_entries_);

        for (const auto& item : db_) {
            write_value<uint64_t>(out, item.first);
            write_value<uint64_t>(out, item.second.size());

            if (!item.second.empty()) {
                out.write(
                    reinterpret_cast<const char*>(item.second.data()),
                    static_cast<std::streamsize>(item.second.size() * sizeof(uint64_t))
                );
                if (!out) {
                    throw std::runtime_error("Failed to write fingerprint binary file");
                }
            }
        }

        changed_ = false;
    }

    void load_binary(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);

        if (!in) {
            throw std::runtime_error("Failed to open fingerprint binary file for reading");
        }

        std::array<char, 8> magic{};
        in.read(magic.data(), magic.size());
        if (!in) {
            throw std::runtime_error("Failed to read fingerprint binary file");
        }

        if (magic != kMagic) {
            throw std::runtime_error("Invalid fingerprint binary magic");
        }

        const uint32_t version = read_value<uint32_t>(in);
        if (version != kVersion) {
            throw std::runtime_error("Unsupported fingerprint binary version");
        }

        const uint64_t unique_address_count = read_value<uint64_t>(in);
        const uint64_t expected_total_entries = read_value<uint64_t>(in);

        std::unordered_map<uint64_t, std::vector<uint64_t>> loaded_db;
        uint64_t loaded_total_entries = 0;

        for (uint64_t i = 0; i < unique_address_count; ++i) {
            const uint64_t address = read_value<uint64_t>(in);
            const uint64_t hash_count = read_value<uint64_t>(in);

            std::vector<uint64_t> hashes(hash_count);
            if (hash_count > 0) {
                in.read(
                    reinterpret_cast<char*>(hashes.data()),
                    static_cast<std::streamsize>(hash_count * sizeof(uint64_t))
                );
                if (!in) {
                    throw std::runtime_error("Failed to read fingerprint binary file");
                }
            }

            loaded_total_entries += hash_count;
            loaded_db[address] = std::move(hashes);
        }

        if (loaded_total_entries != expected_total_entries) {
            throw std::runtime_error("Fingerprint binary total entry count mismatch");
        }

        db_ = std::move(loaded_db);
        total_entries_ = loaded_total_entries;
        changed_ = false;
    }

    bool changed() const {
        return changed_;
    }

    void set_changed(bool changed) {
        changed_ = changed;
    }

private:
    std::unordered_map<uint64_t, std::vector<uint64_t>> db_;
    uint64_t total_entries_ = 0;
    bool changed_ = false;
};

PYBIND11_MODULE(_fingerprint_index, m) {
    py::class_<FingerprintIndex>(m, "FingerprintIndex")
        .def(py::init<>())
        .def("insert", &FingerprintIndex::insert)
        .def("insert_many", &FingerprintIndex::insert_many)
        .def("lookup", &FingerprintIndex::lookup)
        .def("lookup_flat_batch", &FingerprintIndex::lookup_flat_batch)
        .def("contains", &FingerprintIndex::contains)
        .def("remove", &FingerprintIndex::remove)
        .def("clear", &FingerprintIndex::clear)
        .def("size", &FingerprintIndex::size)
        .def("unique_addresses", &FingerprintIndex::unique_addresses)
        .def("items", &FingerprintIndex::items)
        .def("load_items", &FingerprintIndex::load_items)
        .def("save_binary", &FingerprintIndex::save_binary)
        .def("load_binary", &FingerprintIndex::load_binary)
        .def("changed", &FingerprintIndex::changed)
        .def("set_changed", &FingerprintIndex::set_changed);
}

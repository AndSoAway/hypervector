/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2
 * (the "License") found in the LICENSE file in the root directory of this
 * source tree.
 */

#include <factory/index_config.h>
#include <utils/log/exception.h>

#include <charconv>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "factory/name_utils.h"

namespace hypervec {

namespace {

constexpr int kMaxJsonDepth = 64;

struct JsonValue {
  enum class Type {
    Null,
    Boolean,
    Number,
    String,
    Object,
    Array,
  };

  Type type = Type::Null;
  bool boolean = false;
  std::string text;
  std::map<std::string, JsonValue> object;
  std::vector<JsonValue> array;
};

class JsonParser {
 public:
  explicit JsonParser(std::string_view input) : input_(input) {}

  JsonValue Parse() {
    if (input_.size() >= 3 && static_cast<unsigned char>(input_[0]) == 0xEF &&
        static_cast<unsigned char>(input_[1]) == 0xBB &&
        static_cast<unsigned char>(input_[2]) == 0xBF) {
      position_ = 3;
    }
    SkipWhitespace();
    if (position_ == input_.size()) {
      Error("input is empty");
    }
    JsonValue value = ParseValue(0);
    SkipWhitespace();
    if (position_ != input_.size()) {
      Error("unexpected trailing content");
    }
    return value;
  }

 private:
  [[noreturn]] void Error(const char* message) const {
    throw HypervecException("index config JSON error at byte " +
                            std::to_string(position_) + ": " + message);
  }

  void SkipWhitespace() {
    while (position_ < input_.size()) {
      const char ch = input_[position_];
      if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') {
        break;
      }
      ++position_;
    }
  }

  bool Consume(char expected) {
    if (position_ < input_.size() && input_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  void Expect(char expected, const char* message) {
    if (!Consume(expected)) {
      Error(message);
    }
  }

  bool ConsumeKeyword(std::string_view keyword) {
    if (input_.substr(position_, keyword.size()) != keyword) {
      return false;
    }
    position_ += keyword.size();
    return true;
  }

  JsonValue ParseValue(int depth) {
    if (depth > kMaxJsonDepth) {
      Error("nesting is too deep");
    }
    SkipWhitespace();
    if (position_ == input_.size()) {
      Error("expected a JSON value");
    }

    switch (input_[position_]) {
      case '{':
        return ParseObject(depth + 1);
      case '[':
        return ParseArray(depth + 1);
      case '"': {
        JsonValue value;
        value.type = JsonValue::Type::String;
        value.text = ParseString();
        return value;
      }
      case 't': {
        if (!ConsumeKeyword("true")) {
          Error("invalid literal");
        }
        JsonValue value;
        value.type = JsonValue::Type::Boolean;
        value.boolean = true;
        return value;
      }
      case 'f': {
        if (!ConsumeKeyword("false")) {
          Error("invalid literal");
        }
        JsonValue value;
        value.type = JsonValue::Type::Boolean;
        return value;
      }
      case 'n': {
        if (!ConsumeKeyword("null")) {
          Error("invalid literal");
        }
        return {};
      }
      default:
        return ParseNumber();
    }
  }

  JsonValue ParseObject(int depth) {
    Expect('{', "expected an object");
    JsonValue value;
    value.type = JsonValue::Type::Object;
    SkipWhitespace();
    if (Consume('}')) {
      return value;
    }

    while (true) {
      SkipWhitespace();
      if (position_ == input_.size() || input_[position_] != '"') {
        Error("object keys must be strings");
      }
      std::string key = ParseString();
      SkipWhitespace();
      Expect(':', "expected ':' after an object key");
      JsonValue child = ParseValue(depth);
      if (!value.object.emplace(std::move(key), std::move(child)).second) {
        Error("duplicate object key");
      }
      SkipWhitespace();
      if (Consume('}')) {
        return value;
      }
      Expect(',', "expected ',' or '}' in object");
    }
  }

  JsonValue ParseArray(int depth) {
    Expect('[', "expected an array");
    JsonValue value;
    value.type = JsonValue::Type::Array;
    SkipWhitespace();
    if (Consume(']')) {
      return value;
    }

    while (true) {
      value.array.push_back(ParseValue(depth));
      SkipWhitespace();
      if (Consume(']')) {
        return value;
      }
      Expect(',', "expected ',' or ']' in array");
    }
  }

  static int HexDigit(char ch) {
    if (ch >= '0' && ch <= '9') {
      return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f') {
      return ch - 'a' + 10;
    }
    if (ch >= 'A' && ch <= 'F') {
      return ch - 'A' + 10;
    }
    return -1;
  }

  uint32_t ParseCodeUnit() {
    if (input_.size() - position_ < 4) {
      Error("incomplete Unicode escape");
    }
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      const int digit = HexDigit(input_[position_++]);
      if (digit < 0) {
        Error("invalid Unicode escape");
      }
      value = (value << 4) | static_cast<uint32_t>(digit);
    }
    return value;
  }

  static void AppendUtf8(uint32_t code_point, std::string* output) {
    if (code_point <= 0x7F) {
      output->push_back(static_cast<char>(code_point));
    } else if (code_point <= 0x7FF) {
      output->push_back(static_cast<char>(0xC0 | (code_point >> 6)));
      output->push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    } else if (code_point <= 0xFFFF) {
      output->push_back(static_cast<char>(0xE0 | (code_point >> 12)));
      output->push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
      output->push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    } else {
      output->push_back(static_cast<char>(0xF0 | (code_point >> 18)));
      output->push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
      output->push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
      output->push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    }
  }

  void ParseUnicodeEscape(std::string* output) {
    uint32_t code_point = ParseCodeUnit();
    if (code_point >= 0xD800 && code_point <= 0xDBFF) {
      if (!Consume('\\') || !Consume('u')) {
        Error("high surrogate must be followed by a low surrogate");
      }
      const uint32_t low = ParseCodeUnit();
      if (low < 0xDC00 || low > 0xDFFF) {
        Error("invalid low surrogate");
      }
      code_point = 0x10000 + ((code_point - 0xD800) << 10) + (low - 0xDC00);
    } else if (code_point >= 0xDC00 && code_point <= 0xDFFF) {
      Error("unexpected low surrogate");
    }
    AppendUtf8(code_point, output);
  }

  std::string ParseString() {
    Expect('"', "expected a string");
    std::string output;
    while (position_ < input_.size()) {
      const unsigned char ch = input_[position_++];
      if (ch == '"') {
        return output;
      }
      if (ch < 0x20) {
        Error("unescaped control character in string");
      }
      if (ch != '\\') {
        output.push_back(static_cast<char>(ch));
        continue;
      }
      if (position_ == input_.size()) {
        Error("incomplete string escape");
      }
      switch (input_[position_++]) {
        case '"':
          output.push_back('"');
          break;
        case '\\':
          output.push_back('\\');
          break;
        case '/':
          output.push_back('/');
          break;
        case 'b':
          output.push_back('\b');
          break;
        case 'f':
          output.push_back('\f');
          break;
        case 'n':
          output.push_back('\n');
          break;
        case 'r':
          output.push_back('\r');
          break;
        case 't':
          output.push_back('\t');
          break;
        case 'u':
          ParseUnicodeEscape(&output);
          break;
        default:
          Error("invalid string escape");
      }
    }
    Error("unterminated string");
  }

  JsonValue ParseNumber() {
    const size_t begin = position_;
    Consume('-');
    if (position_ == input_.size()) {
      Error("incomplete number");
    }

    if (Consume('0')) {
      if (position_ < input_.size() && input_[position_] >= '0' &&
          input_[position_] <= '9') {
        Error("leading zeros are not allowed");
      }
    } else {
      if (input_[position_] < '1' || input_[position_] > '9') {
        Error("expected a JSON value");
      }
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
    }

    if (Consume('.')) {
      const size_t fraction_begin = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
      if (position_ == fraction_begin) {
        Error("fraction requires at least one digit");
      }
    }

    if (position_ < input_.size() &&
        (input_[position_] == 'e' || input_[position_] == 'E')) {
      ++position_;
      if (position_ < input_.size() &&
          (input_[position_] == '+' || input_[position_] == '-')) {
        ++position_;
      }
      const size_t exponent_begin = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
      if (position_ == exponent_begin) {
        Error("exponent requires at least one digit");
      }
    }

    JsonValue value;
    value.type = JsonValue::Type::Number;
    value.text = std::string(input_.substr(begin, position_ - begin));
    return value;
  }

  std::string_view input_;
  size_t position_ = 0;
};

const std::map<std::string, JsonValue>& RequireObject(const JsonValue& value,
                                                      const char* context) {
  if (value.type != JsonValue::Type::Object) {
    throw HypervecException(std::string(context) + " must be an object");
  }
  return value.object;
}

const JsonValue* FindField(const std::map<std::string, JsonValue>& object,
                           std::string_view name) {
  const auto field = object.find(std::string(name));
  return field == object.end() ? nullptr : &field->second;
}

const JsonValue& RequireField(const std::map<std::string, JsonValue>& object,
                              std::string_view name) {
  const JsonValue* value = FindField(object, name);
  if (value == nullptr) {
    throw HypervecException("missing required index config field '" +
                            std::string(name) + "'");
  }
  return *value;
}

void RejectUnknownFields(const std::map<std::string, JsonValue>& object,
                         std::initializer_list<std::string_view> allowed,
                         const char* context) {
  for (const auto& field : object) {
    bool known = false;
    for (const std::string_view candidate : allowed) {
      if (field.first == candidate) {
        known = true;
        break;
      }
    }
    if (!known) {
      throw HypervecException(std::string("unknown field '") + field.first +
                              "' in " + context);
    }
  }
}

std::string ReadString(const JsonValue& value, const char* context) {
  if (value.type != JsonValue::Type::String) {
    throw HypervecException(std::string(context) + " must be a string");
  }
  return value.text;
}

int64_t ReadInteger(const JsonValue& value, const char* context) {
  if (value.type != JsonValue::Type::Number ||
      value.text.find_first_of(".eE") != std::string::npos) {
    throw HypervecException(std::string(context) + " must be an integer");
  }
  int64_t output = 0;
  const char* begin = value.text.data();
  const char* end = begin + value.text.size();
  const auto result = std::from_chars(begin, end, output);
  if (result.ec != std::errc() || result.ptr != end) {
    throw HypervecException(std::string(context) +
                            " is outside the signed 64-bit range");
  }
  return output;
}

int ReadInt(const JsonValue& value, const char* context) {
  const int64_t integer = ReadInteger(value, context);
  if (integer < std::numeric_limits<int>::min() ||
      integer > std::numeric_limits<int>::max()) {
    throw HypervecException(std::string(context) + " is outside the int range");
  }
  return static_cast<int>(integer);
}

float ReadFloat(const JsonValue& value, const char* context) {
  if (value.type != JsonValue::Type::Number) {
    throw HypervecException(std::string(context) + " must be a number");
  }
  float output = 0.0F;
  const char* begin = value.text.data();
  const char* end = begin + value.text.size();
  const auto result =
      std::from_chars(begin, end, output, std::chars_format::general);
  if (result.ec != std::errc() || result.ptr != end || !std::isfinite(output)) {
    throw HypervecException(std::string(context) +
                            " is outside the finite float range");
  }
  return output;
}

bool ReadBoolean(const JsonValue& value, const char* context) {
  if (value.type != JsonValue::Type::Boolean) {
    throw HypervecException(std::string(context) + " must be a boolean");
  }
  return value.boolean;
}

}  // namespace

MetricType ParseMetricType(std::string_view name) {
  const std::string metric = factory_detail::NormalizeToken(name);
  if (metric == "l2" || metric == "euclidean") {
    return kMetricL2;
  }
  if (metric == "ip" || metric == "inner_product") {
    return kMetricInnerProduct;
  }
  if (metric == "l1" || metric == "manhattan") {
    return kMetricL1;
  }
  if (metric == "linf" || metric == "chebyshev") {
    return kMetricLinf;
  }
  if (metric == "lp") {
    return kMetricLp;
  }
  if (metric == "canberra") {
    return kMetricCanberra;
  }
  if (metric == "bray_curtis") {
    return kMetricBrayCurtis;
  }
  if (metric == "jensen_shannon") {
    return kMetricJensenShannon;
  }
  if (metric == "jaccard") {
    return kMetricJaccard;
  }
  if (metric == "nan_euclidean") {
    return kMetricNaNEuclidean;
  }
  if (metric == "gower") {
    return kMetricGower;
  }
  if (metric == "cosine") {
    HYPERVEC_THROW_MSG(
        "cosine requires an explicit normalization transform, which is not "
        "available in the current factory");
  }
  HYPERVEC_THROW_FMT("unsupported metric '%.*s'", static_cast<int>(name.size()),
                     name.data());
}

std::string_view MetricTypeName(MetricType metric) {
  switch (metric) {
    case kMetricInnerProduct:
      return "inner_product";
    case kMetricL2:
      return "l2";
    case kMetricL1:
      return "l1";
    case kMetricLinf:
      return "linf";
    case kMetricLp:
      return "lp";
    case kMetricCanberra:
      return "canberra";
    case kMetricBrayCurtis:
      return "bray_curtis";
    case kMetricJensenShannon:
      return "jensen_shannon";
    case kMetricJaccard:
      return "jaccard";
    case kMetricNaNEuclidean:
      return "nan_euclidean";
    case kMetricGower:
      return "gower";
    default:
      HYPERVEC_THROW_MSG("unsupported metric enum value");
  }
}

IndexConfig ParseIndexConfig(std::string_view json) {
  const JsonValue root = JsonParser(json).Parse();
  const auto& object = RequireObject(root, "index config");
  RejectUnknownFields(object,
                      {"type", "dimension", "metric", "metric_arg", "backend",
                       "storage", "build", "defaults", "id_mode"},
                      "index config");

  IndexConfig config;
  config.type = ReadString(RequireField(object, "type"), "type");
  config.dimension =
      ReadInteger(RequireField(object, "dimension"), "dimension");
  if (const JsonValue* metric = FindField(object, "metric")) {
    config.metric = ParseMetricType(ReadString(*metric, "metric"));
  }
  if (const JsonValue* metric_arg = FindField(object, "metric_arg")) {
    config.metric_arg = ReadFloat(*metric_arg, "metric_arg");
  }
  if (const JsonValue* backend = FindField(object, "backend")) {
    config.backend = ReadString(*backend, "backend");
  }
  if (const JsonValue* id_mode = FindField(object, "id_mode")) {
    const std::string mode =
        factory_detail::NormalizeToken(ReadString(*id_mode, "id_mode"));
    if (mode == "sequential") {
      config.id_mode = IdMode::Sequential;
    } else if (mode == "external") {
      config.id_mode = IdMode::External;
    } else {
      throw HypervecException("id_mode must be 'sequential' or 'external'");
    }
  }

  if (const JsonValue* storage = FindField(object, "storage")) {
    const auto& fields = RequireObject(*storage, "storage");
    RejectUnknownFields(fields, {"type", "pq_m", "nlocal", "nbits"}, "storage");
    if (const JsonValue* type = FindField(fields, "type")) {
      config.storage.type = ReadString(*type, "storage.type");
    }
    if (const JsonValue* pq_m = FindField(fields, "pq_m")) {
      config.storage.pq_m = ReadInteger(*pq_m, "storage.pq_m");
    }
    if (const JsonValue* nlocal = FindField(fields, "nlocal")) {
      config.storage.nlocal = ReadInteger(*nlocal, "storage.nlocal");
    }
    if (const JsonValue* nbits = FindField(fields, "nbits")) {
      config.storage.nbits = ReadInt(*nbits, "storage.nbits");
    }
  }

  if (const JsonValue* build = FindField(object, "build")) {
    const auto& fields = RequireObject(*build, "build");
    RejectUnknownFields(fields,
                        {"nlist", "graph_degree", "ef_construction",
                         "by_residual", "use_precomputed_table"},
                        "build");
    if (const JsonValue* nlist = FindField(fields, "nlist")) {
      config.build.nlist = ReadInteger(*nlist, "build.nlist");
    }
    if (const JsonValue* degree = FindField(fields, "graph_degree")) {
      config.build.graph_degree = ReadInt(*degree, "build.graph_degree");
    }
    if (const JsonValue* ef = FindField(fields, "ef_construction")) {
      config.build.ef_construction = ReadInt(*ef, "build.ef_construction");
    }
    if (const JsonValue* residual = FindField(fields, "by_residual")) {
      config.build.by_residual = ReadBoolean(*residual, "build.by_residual");
    }
    if (const JsonValue* table = FindField(fields, "use_precomputed_table")) {
      config.build.use_precomputed_table =
          ReadInt(*table, "build.use_precomputed_table");
    }
  }

  if (const JsonValue* defaults = FindField(object, "defaults")) {
    const auto& fields = RequireObject(*defaults, "defaults");
    RejectUnknownFields(fields, {"nprobe", "ef_search"}, "defaults");
    if (const JsonValue* nprobe = FindField(fields, "nprobe")) {
      config.defaults.nprobe = ReadInteger(*nprobe, "defaults.nprobe");
    }
    if (const JsonValue* ef = FindField(fields, "ef_search")) {
      config.defaults.ef_search = ReadInt(*ef, "defaults.ef_search");
    }
  }

  return config;
}

}  // namespace hypervec

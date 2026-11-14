/*
 * Copyright (c) 2024 HyperVec Authors. All rights reserved.
 *
 * This source code is licensed under the Mulan Permissive Software License v2 (the "License") found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <utils/config/runtime_config.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace hypervec {
namespace {

std::string Render(bool value) { return value ? "true" : "false"; }

std::string Render(int value) { return std::to_string(value); }

// Returns the metadata entry for `section.key`, failing the test when absent.
const ConfigOption& FindOption(const char* section, const char* key) {
    for (const auto& option : GetConfigOptions()) {
        if (std::string(section) == option.section &&
            std::string(key) == option.key) {
            return option;
        }
    }
    static const ConfigOption missing{
        "", "", ConfigValueType::kString, "", "", false};
    ADD_FAILURE() << "missing config option " << section << "." << key;
    return missing;
}

// Splits on '\n'; a single trailing '\n' does not yield a trailing empty line.
std::vector<std::string> SplitLines(const std::string& text) {
    std::vector<std::string> lines;
    std::string current;
    for (char ch : text) {
        if (ch != '\n') {
            current.push_back(ch);
            continue;
        }
        lines.push_back(current);
        current.clear();
    }
    if (!current.empty()) {
        lines.push_back(current);
    }
    return lines;
}

bool IsSectionHeader(const std::string& line) {
    return line.size() >= 2 && line.front() == '[' && line.back() == ']';
}

}  // namespace

// The metadata table must not drift from the typed model: every default_value
// is the rendering of the matching HypervecConfig field.
TEST(RuntimeConfig, MetadataDefaultsMatchTypedConfig) {
    const HypervecConfig defaults = DefaultRuntimeConfig();
    ASSERT_EQ(GetConfigOptions().size(), 8U);

    EXPECT_EQ(FindOption("server", "data_root").default_value,
              defaults.server.data_root);
    EXPECT_EQ(FindOption("server", "host").default_value,
              defaults.server.host);
    EXPECT_EQ(FindOption("server", "port").default_value,
              Render(defaults.server.port));

    EXPECT_EQ(FindOption("logging", "enable_logging").default_value,
              Render(defaults.logging.enable_logging));
    EXPECT_EQ(FindOption("logging", "log_level").default_value,
              defaults.logging.log_level);
    EXPECT_EQ(FindOption("logging", "log_to_stderr").default_value,
              Render(defaults.logging.log_to_stderr));
    EXPECT_EQ(FindOption("logging", "log_to_file").default_value,
              Render(defaults.logging.log_to_file));
    EXPECT_EQ(FindOption("logging", "log_file_path").default_value,
              defaults.logging.log_file_path);
}

// Each declared type must match the literal carrying it, so a later parser can
// convert the value without special cases.
TEST(RuntimeConfig, ValueTypesMatchDefaultLiterals) {
    for (const auto& option : GetConfigOptions()) {
        const std::string where =
            std::string(option.section) + "." + option.key;
        switch (option.type) {
            case ConfigValueType::kBool:
                EXPECT_TRUE(option.default_value == "true" ||
                            option.default_value == "false")
                    << where << " = " << option.default_value;
                break;
            case ConfigValueType::kInt: {
                std::size_t consumed = 0;
                EXPECT_NO_THROW(std::stoi(option.default_value, &consumed))
                    << where << " = " << option.default_value;
                EXPECT_EQ(consumed, option.default_value.size())
                    << where << " = " << option.default_value;
                break;
            }
            case ConfigValueType::kString:
                EXPECT_FALSE(option.default_value.empty() && !option.optional)
                    << where << " is empty but not optional";
                break;
        }
    }
}

TEST(RuntimeConfig, OptionMetadataIsNonEmptyAndUnique) {
    for (const auto& option : GetConfigOptions()) {
        EXPECT_GT(std::string(option.section).size(), 0U);
        EXPECT_GT(std::string(option.key).size(), 0U);
        EXPECT_GT(std::string(option.description).size(), 0U);
    }
    for (std::size_t i = 0; i < GetConfigOptions().size(); ++i) {
        for (std::size_t j = i + 1; j < GetConfigOptions().size(); ++j) {
            const ConfigOption& a = GetConfigOptions()[i];
            const ConfigOption& b = GetConfigOptions()[j];
            EXPECT_FALSE(std::string(a.section) == b.section &&
                         std::string(a.key) == b.key)
                << "duplicate option " << a.section << "." << a.key;
        }
    }
}

// The renderer emits options in table order and flushes a section header only
// on a section change, so all options of one section must be contiguous.
TEST(RuntimeConfig, OptionsOfOneSectionAreContiguous) {
    std::vector<std::string> section_order;
    for (const auto& option : GetConfigOptions()) {
        const std::string section = option.section;
        if (section_order.empty() || section_order.back() != section) {
            const bool seen_before =
                std::find(section_order.begin(), section_order.end(),
                          section) != section_order.end();
            EXPECT_FALSE(seen_before)
                << "section " << section
                << " is split into non-contiguous runs";
            section_order.push_back(section);
        }
    }
    ASSERT_EQ(section_order.size(), 2U);
    EXPECT_EQ(section_order[0], "server");
    EXPECT_EQ(section_order[1], "logging");
}

TEST(RuntimeConfig, SampleConfigRendersEveryOption) {
    const std::string sample = RenderSampleConfig();
    ASSERT_FALSE(sample.empty());
    EXPECT_EQ(sample.back(), '\n');
    EXPECT_FALSE(sample.size() >= 2 && sample[sample.size() - 2] == '\n')
        << "sample must not end with a blank line";

    for (const auto& option : GetConfigOptions()) {
        const std::string line =
            std::string(option.key) + " = " + option.default_value;
        EXPECT_NE(sample.find(line), std::string::npos)
            << "missing rendered line: " << line;
        EXPECT_NE(sample.find(std::string("# ") + option.description),
                  std::string::npos)
            << "missing description for " << option.section << "."
            << option.key;
    }
}

TEST(RuntimeConfig, SampleConfigGroupsSectionsWithOneBlankLine) {
    const std::vector<std::string> lines = SplitLines(RenderSampleConfig());

    std::vector<std::size_t> headers;
    std::vector<std::size_t> blanks;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (IsSectionHeader(lines[i])) {
            headers.push_back(i);
        } else if (lines[i].empty()) {
            blanks.push_back(i);
        }
    }

    ASSERT_EQ(headers.size(), 2U);
    EXPECT_EQ(lines[headers[0]], "[server]");
    EXPECT_EQ(lines[headers[1]], "[logging]");
    EXPECT_EQ(headers[0], 0U) << "sample must start with a section header";

    // Exactly one blank line, immediately before the second section.
    ASSERT_EQ(blanks.size(), 1U);
    EXPECT_EQ(blanks[0] + 1, headers[1]);

    // Every rendered key belongs to the section opened right above it.
    std::string current_section;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (IsSectionHeader(lines[i])) {
            current_section = lines[i].substr(1, lines[i].size() - 2);
            continue;
        }
        if (lines[i].empty() || lines[i][0] == '#') {
            continue;
        }
        const std::string key = lines[i].substr(0, lines[i].find(" = "));
        EXPECT_EQ(FindOption(current_section.c_str(), key.c_str()).section,
                  current_section)
            << "key rendered under the wrong section: " << key;
    }
}

}  // namespace hypervec

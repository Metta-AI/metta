#include <gtest/gtest.h>

TEST(CIFailure, IntentionalFailure) {
    FAIL() << "intentional cpp failure for CI visibility";
}

import os

import pytest


@pytest.mark.skipif(
    not os.environ.get("TRAIN_TEST_VENDOR_SDK"), reason="requires the pinned native vendor SDKs"
)
def test_vendor_sdk(vendor_sdk):
    vendor_sdk("baseten")

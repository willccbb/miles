import base64
import io

from PIL import Image

from miles.utils.processing_utils import encode_image_for_rollout_engine


def test_encode_image_preserves_sglang_string_sources():
    source = "https://example.test/image.png"

    assert encode_image_for_rollout_engine(source) == source


def test_encode_image_accepts_encoded_bytes_and_normalizes_to_rgb_png():
    source = io.BytesIO()
    Image.new("RGBA", (2, 3), (255, 0, 0, 127)).save(source, format="PNG")

    encoded = encode_image_for_rollout_engine(source.getvalue())

    assert encoded.startswith("data:image/png;base64,")
    decoded = Image.open(io.BytesIO(base64.b64decode(encoded.split(",", 1)[1])))
    assert decoded.mode == "RGB"
    assert decoded.size == (2, 3)

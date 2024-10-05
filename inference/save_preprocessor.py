from transformers import (
    SegformerImageProcessor,
    SegformerConfig,
    SegformerForSemanticSegmentation,
)
test_image_processor = SegformerImageProcessor.from_pretrained("nvidia/MiT-b0")
test_image_processor.save_pretrained("./sip")


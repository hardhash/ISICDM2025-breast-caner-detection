import torch

class Config():
    YOLO_WEIGHT_DIR = r"E:\ISICDM2025\yolo_weight"                   # YOLO weight path
    CLS_WEIGHT_DIR = r"E:\ISICDM2025\cls_weight"                     # EfficientNet weight path
    TEST_IMG_DIR = r"E:\ISICDM2025\ISICDM2025_images_for_test"       # Test data path
    IMG_SIZE = 512                                                   # Image scale
    NUM_CLASSES = 7                                                  # Number of cls
    YOLO_CONF_THRESH = 0.02                                          # Confidence threshold
    csv_path = r"E:\ISICDM2025\infer-result.csv"                     # Infer result path
    image_name = "ISICDM2025_test_005.png"                           # Infer image name
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # CUDA
    cls_model = 'tf_efficientnet_b0.ns_jft_in1k'
    infer_result_path = r"E:\ISICDM2025\infer-result.csv"            # Infer result path
    postprocess_output_path = r"E:\ISICDM2025\infer-result-postprocess.csv"      # Postprocess result save path
    output_json_path = r"E:\ISICDM2025\infer-result-postprocess_coco.json"   # coco result
    visual_gap = 11

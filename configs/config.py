class detectior_config:
    class model:
        imgsz = (640, 640)
        ckg_path = 'best.pt'

    class input:
        inputsz = (720, 1280)

    class show:
        color = (255,0,0)
        label = "apple"
        show_conf = 0.5

    class post:
        conf_thres = 0.6
        iou_thres = 0.45
        bbox = 'xyxy'
    device = 'cuda:0'

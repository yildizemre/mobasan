import pika
from redis import Redis
import numpy as np
import sys
import signal
import json 
import requests
import os
import sys
from dotenv import load_dotenv
from pathlib import Path
import logging
import graypy
# from py_zipkin.zipkin import zipkin_span,create_http_headers_for_new_span , zipkin_client_span
# from py_zipkin.transport import BaseTransportHandler
import cv2
from pydoc import cli
from telethon import TelegramClient, events, sync
from tkinter import *
from tkinter import messagebox
import argparse
import sys
from pathlib import Path
from datetime import datetime
from time import time
import torch
import torch.backends.cudnn as cudnn
import pandas as pd

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import check_dataset, check_requirements, check_yaml, clean_str, segments2boxes, \
    xywh2xyxy, xywhn2xyxy, xyxy2xywhn, xyn2xy
from utils.torch_utils import torch_distributed_zero_first

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams, img2label_paths
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

fire_dataframe_array = []

def send_message(file_path):
   #API details
   user_details = "@hypemanagerai"

   send_time= datetime.now()
   message_content = str(send_time)+" HypeGen.AI - OFFICE CAM 1"
  
   #Raise a warning if no input is given
   if (len(user_details) <=0) & (len(message_content)<=1):
       print("ERROR")
   else:
 #These API codes wont work, hence create your own
       api_id = 1588528
       api_hash = '87e8c2569ffac6001ff62448c366a14d'
       #Initialise telegram client with API codes
       client = TelegramClient('session_name', api_id, api_hash)
       #Start the process
       client.start()
       #Send the message
       client.send_message(user_details, message_content)
       client.send_file("@hypemanagerai",file_path)
       client.disconnect()
env_path = Path('.') / 'project1.env'
load_dotenv(dotenv_path=env_path)

log_level = os.getenv('LOG_LEVEL')
graylog_host = os.getenv('GRAYLOG_HOST')
graylog_port = os.getenv('GRAYLOG_PORT')

logger = logging.getLogger()
def json_load(body):
    json_data = body.decode()
    return json.loads(json_data)

# @zipkin_client_span(service_name=os.getenv('PROJECT_NAME'), span_name='redis get')
def redis_get(json_data):
    if redis.exists(json_data['frame_uuid'] + "_data"):
        return  redis.get(json_data['frame_uuid'] + "_data")
    return False

# @zipkin_client_span(service_name=os.getenv('PROJECT_NAME'), span_name='frame decode')
def frame2(redis_data):
    jpg_as_np = np.frombuffer(redis_data, dtype=np.uint8)
    org_frame = jpg_as_np.reshape(480,848,3)   
    return org_frame

if graylog_host:
    graylog_handler = graypy.GELFUDPHandler(graylog_host, int(graylog_port))
    logger.addHandler(graylog_handler)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logger.setLevel(log_level)

# logger.info('Frame_uploader starting')
logger.setLevel(log_level)
# logger.info('{} starting'.format(os.getenv('PROJECT_NAME')))

def signal_handler(sig, frame):
    connection.close()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOST')))
channel = connection.channel()
# logger.info('ConnectionRabbitMQ')
# logger.info('RabbitMQ Queue'+str(os.getenv('RABBIT_ROUTING_KEY')))
redis = Redis(host=os.getenv('REDIS_HOST'), port=os.getenv('REDIS_PORT'),  db=os.getenv('REDIS_DB_PORT'))
# logger.info('ConnectionRedis')
# class HttpTransport(BaseTransportHandler):

#     def get_max_payload_bytes(self):
#         return None

#     def send(self, zipkin_dict):
#         # The collector expects a thrift-encoded list of spans.
#         requests.post(
#             'http://localhost:9411/api/v1/spans',
#             data=zipkin_dict,
#             headers={'Content-Type': 'application/x-thrift'},
#         )
  
def gelenler(original_frame,imgsz):

    # print(f'image {1}/{2}: ', end='')
    img = letterbox(original_frame, imgsz, stride=32, auto=True)[0]
    
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    
    return 'path', np.ascontiguousarray(img), original_frame, 'cap'
# some_handler = HttpTransport()
@torch.no_grad()
def run(ch,
    properties,
    method,
    body,
    weights=ROOT / 'best.pt',  # model.pt path(s)
    source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
    imgsz=640,  # inference size (pixels)
    conf_thres=0.10,  # confidence threshold
    iou_thres=0.10,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,
    ):
    global fire_dataframe_array
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #     ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)
    json_data = json_load(body)       
    redis_data = redis_get(json_data)
    # print(json_data)
    if redis_data:           
        original_frame = frame2(redis_data)
        # print(type(original_frame))

        img = letterbox(original_frame, imgsz, stride=32, auto=True)[0]
    
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        # print('****************************',type('path'))
        # print('****************************',type(img))
        # print('****************************',type(original_frame))
        # print('****************************',type('cap'))
        # dataset=gelenler(original_frame,640)
        # vid_path, vid_writer = [None] * bs, [None] * bs

        # if True:
        #     bs = len(dataset)  # batch_size
        # vid_path, vid_writer = [None] * bs, [None] * bs
        # Run inference
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
        dt, seen = [0.0, 0.0, 0.0], 0
        # for path, img, im0s, _ in dataset:
        path='path'
        im0s=original_frame
        # print('****************************',type(path))
        # print('****************************',type(img))
        # print('****************************',type(im0s))
        # print('****************************',type())
        t1 = time_sync()
        # print('type**************',type(img))
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, s, im0 = path, '', im0s.copy()
            # p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # if save_txt:  # Write to file
                    #     # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     # line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        if(names[c]) =="fire":
                            print("fire")
                            
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # if save_crop:
                        #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            print(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            # im0 = annotator.result()
            # if view_img:
            #     cv2.imshow('test', im0)
            #     cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if vid_path[i] != save_path:  # new video
            #             vid_path[i] = save_path
            #             if isinstance(vid_writer[i], cv2.VideoWriter):
            #                 vid_writer[i].release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #                 save_path += '.mp4'
            #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer[i].write(im0)

        # Print results
        # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        # # print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        # if save_txt or save_img:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #     print(f"Results saved to {colorstr('bold', save_dir)}{s}")
        # if update:
        #     strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
        im0 = annotator.result()

        try:
            cv2.imshow('Frame',im0)
            cv2.waitKey(1)
        except:
            cv2.imshow('Frame',original_frame)
            cv2.waitKey(1)


        # else:
        #     logger.info("Data Not Found")
        #     ch.basic_ack(delivery_tag=method.delivery_tag)




# @zipkin_client_span(service_name=os.getenv('PROJECT_NAME'), span_name='json load')


def callback(ch,method, properties, body):
    run(ch,properties,method,body)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=os.getenv('RABBIT_ROUTING_KEY'), on_message_callback=callback,auto_ack=True)
channel.start_consuming()

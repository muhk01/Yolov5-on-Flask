import argparse
import paho.mqtt.client as mqtt
import json
from utils.datasets import *
from utils.utils import *

MQTT_TOPIC = "v1/devices/me/telemetry"
mylist = []
mycount = []
broker_url = "demo.thingsboard.io"
broker_port = 1883

username = "aTbDLqllhSQAyWLeE6rx"
password = '' 

client = mqtt.Client()
client.username_pw_set(username)
client.connect(broker_url, broker_port)
client.loop_start()


def detect(source):
	out, weights, half, view_img, save_txt, imgsz = \
		opt.output,  opt.weights, opt.half, opt.view_img, opt.save_txt, opt.img_size
	
	device = torch_utils.select_device(opt.device)
	if os.path.exists(out):
		shutil.rmtree(out)  # delete output folder
	os.makedirs(out)  # make new output folder

	# Load model
	google_utils.attempt_download(weights)
	model = torch.load(weights, map_location=device)['model']
	
	model.to(device).eval()

	# Second-stage classifier
	classify = False
	if classify:
		modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
		modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
		modelc.to(device).eval()

	# Half precision
	half = half and device.type != 'cpu'  # half precision only supported on CUDA
	print('half = ' + str(half))
	print('augment = ' + str(opt.augment))
	print(opt.conf_thres)
	print(opt.iou_thres)
	print(opt.classes)
	print(opt.agnostic_nms)
		
	if half:
		model.half()

	# Set Dataloader
	vid_path, vid_writer = None, None
	dataset = LoadImages(source, img_size=imgsz)
	#dataset = LoadStreams(source, img_size=imgsz)
	names = model.names if hasattr(model, 'names') else model.modules.names
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

	# Run inference
	t0 = time.time()
	classSend = []
	countSend = []
	img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
	_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
	for path, img, im0s, vid_cap in dataset:
		img = torch.from_numpy(img).to(device)
		img = img.half() if half else img.float()  # uint8 to fp16/32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		# Inference
		t1 = torch_utils.time_synchronized()
		pred = model(img, augment=opt.augment)[0]
		
		# Apply NMS
		pred = non_max_suppression(pred, 0.4, 0.5,
							   fast=True, classes=None, agnostic=False)
		t2 = torch_utils.time_synchronized()

		# Apply Classifier
		if classify:
			pred = apply_classifier(pred, modelc, img, im0s)

		for i, det in enumerate(pred):  # detections per image
			p, s, im0 = path, '', im0s

			save_path = str(Path(out) / Path(p).name)
			s += '%gx%g ' % img.shape[2:]  # print string
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
			if det is not None and len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum()  # detections per class
					s += '%g %s, ' % (n, names[int(c)])  # add to string
					listDet = ['person','bicycle','car','motorbike','bus','truck','bird','cat','dog','horse','cow','backpack','umbrella','handbag','kite','cell phone']
					
					if(str(names[int(c)]) in listDet):
						countSend.append('%s' % (names[int(c)]))
						classSend.append('%g' % (n))

				for *xyxy, conf, cls in det:
					label = '%s %.2f' % (names[int(cls)], conf)
					plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

			# Print time (inference + NMS)
			#print('/nIsi: ')
			#print(classSend)
			#print(countSend)
			#print('/n')
			data_set = {"Object": classSend, "Count": countSend}
			MQTT_MSG = json.dumps(data_set)
			client.publish(MQTT_TOPIC, MQTT_MSG)
			print('%sDone. (%.3fs)' % (s, t2 - t1))
			del classSend[:]
			del countSend[:]
			im0 = cv2.resize(im0,(800,600))
			cv2.imshow(p, im0)
			if cv2.waitKey(1) == ord('q'):  # q to quit
				raise StopIteration

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
	#parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
	parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
	parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
	parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
	parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
	parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
	parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
	parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--view-img', action='store_true', help='display results')
	parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
	parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
	parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
	parser.add_argument('--augment', action='store_true', help='augmented inference')
	opt = parser.parse_args()
	print(opt)

	with torch.no_grad():
		detect('4.mp4')

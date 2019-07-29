import cv2
import argparse
import mtcnn
import os


def cmd_args():
    parser = argparse.ArgumentParser(description='Demo with real-time face and facial keypoints detection')
    parser.add_argument('--input', type=str, 
                        default=os.environ.get('INPUT_VIDEO', ''), 
                        help="Read from a video or stream")
    parser.add_argument('--save', type=str, 
                        default=eval(os.environ.get('OUTPUT_VIDEO', 'None')),
                        help="If set, Save as video. Else show it on screen.")
    parser.add_argument("--minsize", type=int, 
                        default=int(os.environ.get('FACE_BOX_MINSIZE', 100)),
                        help="Min size of faces you want to detect. Larger number will speed up detect method.")
    parser.add_argument("--device", type=str, 
                        default=os.environ.get('GPU_DEVICE', 'cpu'),
                        help="Target device to process video.")
    parser.add_argument('--confidence', type=float, 
                        default=float(os.environ.get('FACE_CONFIDENCE', 0.9)),
                        help='Face confidence for the last output network')
    parser.add_argument('--model_dir', type=str, default='',
                        help='Path to cascade networks')
    parser.add_argument('--codec', type=str, 
                        default=os.environ.get("OUT_CODEC", "XVID"),
                        help='codec to use with opencv')
    return parser.parse_args()


def main(args):
    assert args.input != '', 'Provide input video either as env variable INPUT_VIDEO or in command line as --input'
    if args.model_dir == '':
        pnet, rnet, onet = mtcnn.get_net_caffe('FaceDetector/output/caffe_models/')
    else:
        pnet, rnet, onet = mtcnn.get_net_caffe(args.model_dir)

    detector = mtcnn.FaceDetector(pnet, rnet, onet, device=args.device)
    
    cap = cv2.VideoCapture(args.input)
    print('Connected to', args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    if args.save is not None:
        fourcc = cv2.VideoWriter_fourcc(*args.codec)
        out = cv2.VideoWriter(args.save, fourcc, fps, size)
    
    while True:    
        res, image = cap.read()
        if not res:
            break
    
        boxes, landmarks = detector.detect(image, minsize=args.minsize, 
                                           threshold=[0.6,0.7,args.confidence])
        image = mtcnn.utils.draw.draw_boxes2(image, boxes)
        image = mtcnn.utils.draw.batch_draw_landmarks(image, landmarks)
        
        if args.save is None:
            cv2.imshow("demo", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            out.write(image)
    
    cap.release()
    cv2.destroyAllWindows()
    if args.save is not None:
        out.release()
        
        
if __name__ == '__main__':
    args = cmd_args()
    main(args)

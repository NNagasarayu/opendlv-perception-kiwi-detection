#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <stdint.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <chrono>

const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
"{image i        |<none>| input image   }"
"{video v       |<none>| input video   }"
"{device d       |<cpu>| input device   }"
;

using namespace cv;
using namespace dnn;
using namespace std;

// Initialize the parameters
float confThreshold = 0.5f; // Confidence threshold
float nmsThreshold = 0.4f;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs, 
  vector<vector<int>>& centerBottomCoords, const bool& VERBOSE);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);

int32_t main(int32_t argc, char **argv) {
  int32_t retCode{1};
  auto commandlineArguments = cluon::getCommandlineArguments(argc, argv);
  if ( (0 == commandlineArguments.count("cid")) ||
       (0 == commandlineArguments.count("name")) ||
       (0 == commandlineArguments.count("width")) ||
       (0 == commandlineArguments.count("height")) ||
       (0 == commandlineArguments.count("data-path"))) {
    std::cerr << argv[0] << " attaches to a shared memory area containing an ARGB image." << std::endl;
    std::cerr << "Usage:   " << argv[0] << " --cid=<OD4 session> --name=<name of shared memory area> --width=<width of frame> --height=<height of frame> --data-path=<path to folder containing yolo setup> [--verbose]" << std::endl;
    std::cerr << "         --cid:    CID of the OD4Session to send and receive messages" << std::endl;
    std::cerr << "         --name:   name of the shared memory area to attach" << std::endl;
    std::cerr << "         --width:  width of the frame" << std::endl;
    std::cerr << "         --height: height of the frame" << std::endl;
    std::cerr << "         --data-path: path to the data folder with the yolo setup" << std::endl;
    std::cerr << "Example: " << argv[0] << " --cid=111 --name=img.argb --width=1280 --height=720 --data-path=/usr/data --verbose" << std::endl;
  }
  else {
    const std::string NAME{commandlineArguments["name"]};
    const uint32_t WIDTH{static_cast<uint32_t>(
      std::stoi(commandlineArguments["width"]))};
    const uint32_t HEIGHT{static_cast<uint32_t>(
      std::stoi(commandlineArguments["height"]))};
    const std::string DATA_PATH{commandlineArguments["data-path"]};
    const bool VERBOSE{commandlineArguments.count("verbose") != 0};
    
    // Load names of classes
    string classesFile = DATA_PATH + "/yolo.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    string device = "cpu";
    
    // Give the configuration and weight files for the model
    String modelConfiguration = DATA_PATH + "/yolov3_custom_test.cfg";
    String modelWeights = DATA_PATH + "/yolov3_custom_train_final.weights";

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);

    if (device == "cpu")
    {
      if(VERBOSE){
        cout << "Using CPU device" << endl;
      }
      net.setPreferableBackend(DNN_TARGET_CPU);
    }
    else if (device == "gpu")
    {
      if(VERBOSE){
        cout << "Using GPU device" << endl;
      }
      net.setPreferableBackend(DNN_BACKEND_CUDA);
      net.setPreferableTarget(DNN_TARGET_CUDA);
    }
    
    // Attach to the shared memory.
    std::unique_ptr<cluon::SharedMemory> 
      sharedMemory{new cluon::SharedMemory{NAME}};
    if (sharedMemory && sharedMemory->valid()) {
      std::clog << argv[0] << ": Attached to shared memory '" 
        << sharedMemory->name() << " (" << sharedMemory->size() << " bytes)." 
        << std::endl;

      // Interface to a running OpenDaVINCI session; here, 
      // you can send and receive messages.
      cluon::OD4Session od4{static_cast<uint16_t>(
        std::stoi(commandlineArguments["cid"]))};
      
      
      while (od4.isRunning()) {
        cv::Mat img;

        // Wait for a notification of a new frame.
        sharedMemory->wait();

        // Lock the shared memory.
        sharedMemory->lock();
        {
          // Copy image into cvMat structure.
          // Be aware of that any code between lock/unlock is blocking
          // the camera to provide the next frame. Thus, any
          // computationally heavy algorithms should be placed outside
          // lock/unlock
          cv::Mat wrapped(HEIGHT, WIDTH, CV_8UC4, sharedMemory->data());
          img = wrapped.clone();
        }
        sharedMemory->unlock();
        
        //Removes Alpha channel
        cv::cvtColor(img, img, cv::COLOR_RGBA2RGB);
        
        cv::Mat blob;
        
         // Create a 4D blob from a frame.
        blobFromImage(img, blob, 1/255.0, cv::Size(inpWidth, inpHeight), 
          true, false);
        
        //Sets the input to the network
        net.setInput(blob);
        
        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));
        
        // Remove the bounding boxes with low confidence
        vector<vector<int>> centerBottomCoords;
        postprocess(img, outs, centerBottomCoords, VERBOSE);
        
        opendlv::logic::perception::KiwiPosition kiwiMsg;
        cluon::data::TimeStamp sampleTime;
        if (centerBottomCoords.size()>=1) {
          //For H=720 W=1280 range is x->[-640, 640] y->[-120, 600] to keep same
          //coordiante system as in cone detection.
          int x = centerBottomCoords[0][0] - WIDTH/2;
          int y = HEIGHT  - centerBottomCoords[0][1] - HEIGHT/6;
          kiwiMsg.x(x);
          kiwiMsg.y(y);
        } else {
          kiwiMsg.x(0);
          kiwiMsg.y(-1000);
        }
            
        if (VERBOSE) {
          std::cout<<"msg: x=" << kiwiMsg.x()<<", y=" <<kiwiMsg.y()<<std::endl;
        }
        od4.send(kiwiMsg, sampleTime, 0);
        
        if(VERBOSE){
          if (centerBottomCoords.size()>1) {
            std::cout << "more than one kiwi detected" << std::endl;
          }
        
          std::cout << "verbose" << std::endl;
          // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
          vector<double> layersTimes;
          double freq = getTickFrequency() / 1000;
          double t = net.getPerfProfile(layersTimes) / freq;
          string label = format("Inference time for a frame : %.2f ms", t);
          putText(img, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
          
          // Write the frame with the detection boxes
          //Mat detectedFrame;
          //img.convertTo(detectedFrame, CV_8U);
          //if (parser.has("image")) imwrite(outputFile, detectedFrame);
          //else video.write(detectedFrame);
          std::cout << "Show image" << std::endl;
          cv::line(img, cv::Point(0,600), cv::Point(1279,  600), cv::Scalar(255,0,0), 1); //y=0
          cv::line(img, cv::Point(0,550), cv::Point(1279,  550), cv::Scalar(255,0,0), 1); //y=50
          cv::line(img, cv::Point(0,500), cv::Point(1279,  500), cv::Scalar(255,0,0), 1); //y=100
          cv::line(img, cv::Point(0,450), cv::Point(1279,  450), cv::Scalar(255,0,0), 1); //y=150
          cv::line(img, cv::Point(0,400), cv::Point(1279,  400), cv::Scalar(255,0,0), 1); //y=200
          imshow("Detected", img);
          
          cv::waitKey(1);
        }
      }
    }
  }
  return retCode;
}
    
    // Get the video writer initialized to save the output video
    // Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs, 
  vector<vector<int>>& centerBottomCoords, const bool& VERBOSE)
{
  vector<int> classIds;
  vector<float> confidences;
  vector<Rect> boxes;
  vector<vector<int>> allCenterBottomCoords;
  
  for (size_t i = 0; i < outs.size(); ++i)
  {
    // Scan through all the bounding boxes output from the network and 
    // keep only the ones with high confidence scores. Assign the box's class
    // label as the class with the highest score for the box.
    float* data = (float*)outs[i].data;
    for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
    {
      Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
      Point classIdPoint;
      double confidence;
      // Get the value and location of the maximum score
      minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
      if (confidence > confThreshold)
      {
        int centerX = (int)(data[0] * frame.cols);
        int centerY = (int)(data[1] * frame.rows);
        int width = (int)(data[2] * frame.cols);
        int height = (int)(data[3] * frame.rows);
        int left = centerX - width / 2;
        int top = centerY - height / 2;
        int bottom = centerY + height / 2;                
        
        classIds.push_back(classIdPoint.x);
        confidences.push_back((float)confidence);
        boxes.push_back(Rect(left, top, width, height));
        
        allCenterBottomCoords.push_back(vector<int>{centerX, bottom});
      }
    }
  }
  
  // Perform non maximum suppression to eliminate redundant overlapping boxes 
  // with lower confidences
  vector<int> indices;
  NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
  for (size_t i = 0; i < indices.size(); ++i)
  {
    int idx = indices[i];
    centerBottomCoords.push_back(allCenterBottomCoords[idx]); 
    if (VERBOSE) {
      Rect box = boxes[idx];
      drawPred(classIds[idx], confidences[idx], box.x, box.y, 
        box.x + box.width, box.y + box.height, frame);
    }
  }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
  //Draw a rectangle displaying the bounding box
  rectangle(frame, Point(left, top), Point(right, bottom), 
    Scalar(255, 178, 50), 3);
  
  //Get the label for the class name and its confidence
  string label = format("%.2f", conf);
  if (!classes.empty())
  {
    CV_Assert(classId < (int)classes.size());
    label = classes[classId] + ":" + label;
  }
  
  //Display the label at the top of the bounding box
  int baseLine;
  Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  top = max(top, labelSize.height);
  cv::Point p1 = Point(left, top -
    static_cast<int>(round(1.5*labelSize.height)));
  cv::Point p2 = Point(left + static_cast<int>(round(1.5*labelSize.width)), 
    top + baseLine);
  rectangle(frame, p1, p2, Scalar(255, 255, 255), FILLED);
  putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75,
    Scalar(0,0,0),1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
  static vector<String> names;
  if (names.empty())
  {
    // Get the indices of the output layers, i.e. the layers with unconnected
    // outputs
    vector<int> outLayers = net.getUnconnectedOutLayers();
    
    // Get the names of all the layers in the network
    vector<String> layersNames = net.getLayerNames();
    
    // Get the names of the output layers in names
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i)
    names[i] = layersNames[outLayers[i] - 1];
  }
  return names;
}


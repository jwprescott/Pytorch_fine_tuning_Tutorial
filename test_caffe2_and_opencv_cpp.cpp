// Test caffe2 and opencv C++ APIs
// Based on following: https://github.com/leonardvandriel/caffe2_cpp_tutorial/blob/master/src/caffe2/binaries/pretrained.cc
// See CMakeLists.txt

#include <caffe2/core/init.h>
#include <caffe2/core/net.h>
#include <caffe2/core/operator.h>
#include <caffe2/utils/proto_utils.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>

CAFFE2_DEFINE_string(init_net, "/home/prescott/Projects/Pytorch_fine_tuning_Tutorial/init_net.pb",
                     "The given path to the init protobuffer.")
CAFFE2_DEFINE_string(predict_net, "/home/prescott/Projects/Pytorch_fine_tuning_Tutorial/predict_net.pb",
                     "The given path to the predict protobuffer.")
CAFFE2_DEFINE_string(file, "/home/prescott/Projects/Pytorch_fine_tuning_Tutorial/pic_00023075_028.jpg", "The image file.")
// CAFFE2_DEFINE_string(classes, "res/imagenet_classes.txt", "The classes file.");
CAFFE2_DEFINE_int(size, 224, "The image file.")

// CAFFE2_DEFINE_string(init_net, "/home/prescott/Projects/caffe2_cpp_tutorial-master/script/res/squeezenet_init_net.pb",
//                      "The given path to the init protobuffer.");
// CAFFE2_DEFINE_string(predict_net, "/home/prescott/Projects/caffe2_cpp_tutorial-master/script/res/squeezenet_predict_net.pb",
//                      "The given path to the predict protobuffer.");
// CAFFE2_DEFINE_string(file, "/home/prescott/Projects/caffe2_cpp_tutorial-master/script/res/image_file.jpg", "The image file.");
// CAFFE2_DEFINE_string(classes, "/home/prescott/Projects/caffe2_cpp_tutorial-master/script/res/imagenet_classes.txt", "The classes file.");
// CAFFE2_DEFINE_int(size, 227, "The image file.");

namespace caffe2 {

void run() {
  std::cout << std::endl;
  std::cout << "## Caffe2 Loading Pre-Trained Models Tutorial ##" << std::endl;
  std::cout << "https://caffe2.ai/docs/zoo.html" << std::endl;
  std::cout << "https://caffe2.ai/docs/tutorial-loading-pre-trained-models.html"
            << std::endl;
  std::cout << "https://caffe2.ai/docs/tutorial-image-pre-processing.html"
            << std::endl;
  std::cout << std::endl;

  if (!std::ifstream(FLAGS_init_net).good() ||
      !std::ifstream(FLAGS_predict_net).good()) {
    std::cerr << "error: Squeezenet model file missing: "
              << (std::ifstream(FLAGS_init_net).good() ? FLAGS_predict_net
                                                       : FLAGS_init_net)
              << std::endl;
    std::cerr << "Make sure to first run ./script/download_resource.sh"
              << std::endl;
    return;
  }

  if (!std::ifstream(FLAGS_file).good()) {
    std::cerr << "error: Image file missing: " << FLAGS_file << std::endl;
    return;
  }

//   if (!std::ifstream(FLAGS_classes).good()) {
//     std::cerr << "error: Classes file invalid: " << FLAGS_classes << std::endl;
//     return;
//   }

  std::cout << "init-net: " << FLAGS_init_net << std::endl;
  std::cout << "predict-net: " << FLAGS_predict_net << std::endl;
  std::cout << "file: " << FLAGS_file << std::endl;
  std::cout << "size: " << FLAGS_size << std::endl;

  std::cout << std::endl;
  
  // >>> img =
  // skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
  auto image = cv::imread(FLAGS_file);  // CV_8UC3
  std::cout << "image size: " << image.size() << std::endl;

  // scale image to fit
  cv::Size scale(std::max(FLAGS_size * image.cols / image.rows, FLAGS_size),
                 std::max(FLAGS_size, FLAGS_size * image.rows / image.cols));
  cv::resize(image, image, scale);
  std::cout << "scaled size: " << image.size() << std::endl;

  // crop image to fit
  cv::Rect crop((image.cols - FLAGS_size) / 2, (image.rows - FLAGS_size) / 2,
                FLAGS_size, FLAGS_size);
  image = image(crop);
  std::cout << "cropped size: " << image.size() << std::endl;

  // convert to float, normalize to mean 128
  image.convertTo(image, CV_32FC3, 1.0, -128);
  std::cout << "value range: ("
            << *std::min_element((float *)image.datastart,
                                 (float *)image.dataend)
            << ", "
            << *std::max_element((float *)image.datastart,
                                 (float *)image.dataend)
            << ")" << std::endl;

  // convert NHWC to NCHW
  vector<cv::Mat> channels(3);
  cv::split(image, channels);
  std::vector<float> data;
  for (auto &c : channels) {
    data.insert(data.end(), (float *)c.datastart, (float *)c.dataend);
  }
  std::vector<TIndex> dims({1, image.channels(), image.rows, image.cols});
  TensorCPU tensor(dims, data, NULL);

  // Load model
  NetDef init_net, predict_net;
  
  // >>> with open(path_to_INIT_NET) as f:
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net, &init_net));

  // >>> with open(path_to_PREDICT_NET) as f:
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_predict_net, &predict_net));

  init_net.set_name("init_net");
  predict_net.set_name("predict_net");
  
  // >>> p = workspace.Predictor(init_net, predict_net)
  Workspace workspace("tmp");
//  CAFFE_ENFORCE(workspace.RunNetOnce(init_net));
  CAFFE_ENFORCE(workspace.CreateNet(init_net));
  CAFFE_ENFORCE(workspace.RunNet(init_net.name()));
  auto input = workspace.CreateBlob("data")->GetMutable<TensorCPU>();
  input->ResizeLike(tensor);
  input->ShareData(tensor);
//  CAFFE_ENFORCE(workspace.RunNetOnce(predict_net));
  CAFFE_ENFORCE(workspace.CreateNet(predict_net));
  CAFFE_ENFORCE(workspace.RunNet(predict_net.name()));

  // Model output classifier vector
  auto output = workspace.GetBlob("1277")->Get<TensorCPU>();
  
  std::cout << "output: " << output.data<float>()[0] << std::endl;

  // Average pool layer for DenseNet 121
  auto img_avg_pool = workspace.GetBlob("1272")->Get<TensorCPU>();

  vector<string> net_names = workspace.Nets();
  for( int i = 0; i < net_names.size(); ++i )
  {
      std::cout << "Net: " << net_names[i] << std::endl;
  }
  
  // Classifier weights layer for DenseNet121
  auto weights_classifier = workspace.GetNet("init_net")->GetOperators().at(605)->GetRepeatedArgument<float>("values");
  
  // Calculate inner product of average pool layer and classifier weights layer for
  // class activation map
  std::vector<TIndex> dims2({1024, 49});
  img_avg_pool.Resize(dims2);

  std::cout << "Complete" << std::endl;
}

}   // namespace caffe2

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}

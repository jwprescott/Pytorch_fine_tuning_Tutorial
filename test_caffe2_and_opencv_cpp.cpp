// Test caffe2 and opencv C++ APIs
// Based on following: https://github.com/leonardvandriel/caffe2_cpp_tutorial/blob/master/src/caffe2/binaries/pretrained.cc
// See CMakeLists.txt

#include <caffe2/core/init.h>
#include <caffe2/core/net.h>
#include <caffe2/core/operator.h>
#include <caffe2/utils/proto_utils.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Core>

#include <fstream>

CAFFE2_DEFINE_string(init_net, "/home/prescott/Projects/Pytorch_fine_tuning_Tutorial/init_net.pb",
                     "The given path to the init protobuffer.")
CAFFE2_DEFINE_string(predict_net, "/home/prescott/Projects/Pytorch_fine_tuning_Tutorial/predict_net.pb",
                     "The given path to the predict protobuffer.")
CAFFE2_DEFINE_string(file, "/home/prescott/Desktop/output_20180110_162914/test_out_test/1_images_infection/00023068_049.jpg", "The image file.")
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

//  // crop image to fit
//  cv::Rect crop((image.cols - FLAGS_size) / 2, (image.rows - FLAGS_size) / 2,
//                FLAGS_size, FLAGS_size);
//  image = image(crop);
//  std::cout << "cropped size: " << image.size() << std::endl;

  // TODO: UNIT NORMALIZE IMAGES (DIVIDE EACH CHANNEL BY 255), THEN SUBTRACT
  // MEAN AND STD OF IMAGENET
//  // convert to float, normalize to mean 128
  image.convertTo(image, CV_32FC3);
//  image.convertTo(image, CV_32FC3, 1.0, -128);
//  std::cout << "value range: ("
//            << *std::min_element((float *)image.datastart,
//                                 (float *)image.dataend)
//            << ", "
//            << *std::max_element((float *)image.datastart,
//                                 (float *)image.dataend)
//            << ")" << std::endl;

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


  // Average pool layer for DenseNet 121
  auto img_avg_pool = workspace.GetBlob("1272")->Get<TensorCPU>();

  // DEBUG: values from img_avg_pool layer
//  for(int i = 0; i < 49; ++i)
//  {
//      std::cout << "img_avg_pool: " << img_avg_pool.data<float>()[1022*7*7 + i] << std::endl;
//  }

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
  // DEBUG: values from img_avg_pool layer after resize
//  for(int i = 0; i < 49; ++i)
//  {
//      std::cout << "img_avg_pool, resized: " << img_avg_pool.data<float>()[1022*49 + i] << std::endl;
//  }

  const auto &img_avg_pool_data = img_avg_pool.data<float>();

//  Eigen::Matrix<float,1024,49> mat_img_avg_pool;
//   std::unique_ptr<Eigen::Matrix<float,1024,49> > mat_img_avg_pool(new Eigen::Matrix<float,1024,49>);
  Eigen::MatrixXf mat_img_avg_pool(1024,49);
   std::cout << mat_img_avg_pool.rows() << std::endl;
   std::cout << mat_img_avg_pool.cols() << std::endl;
  for (auto i = 0; i < img_avg_pool.size(); i++) {
    mat_img_avg_pool(i) = img_avg_pool_data[i];
  }
  std::cout << std::endl;

//  // DEBUG: Make sure values/indices from img_avg_pool layer and mat_img_avg_pool match
//  for(int i = 0; i < 49; ++i)
//  {
//      std::cout << "img_avg_pool, mat_img_avg_pool: "
//                << img_avg_pool_data[1022*49 + i] << ", "
//                << mat_img_avg_pool(1022*49 + i)
//                << std::endl;
//  }

  // DEBUG
//  for(int i = 0; i < mat_img_avg_pool.rows(); ++i)
//  {
//      for(int j = 0; j < mat_img_avg_pool.cols(); ++j)
//      {
//          std::cout << "row " << i
//                    << ", col " << j
//                    << ", value " << mat_img_avg_pool(i,j)
//                    << std::endl;
//      }
//  }

  Eigen::MatrixXf mat_weights_classifier(1,1024);
  for (auto i = 0; i < weights_classifier.size(); ++i) {
    mat_weights_classifier(i) = weights_classifier[i];
  }
 std::cout << std::endl;

//   // DEBUG: Make sure values/indices from weights_classifier and mat_weights_classifier match
//   for(auto i = 0; i < weights_classifier.size(); ++i)
//   {
//       std::cout << "weights_classifier, mat_weights_classifier: "
//                 << weights_classifier[i] << ", "
//                 << mat_weights_classifier(i)
//                 << std::endl;
//   }

// Eigen::MatrixXf saliency_map(1,49);
 Eigen::MatrixXf saliency_map;
 saliency_map = mat_weights_classifier * mat_img_avg_pool;
// Eigen::MatrixXf saliency_map(1,49);
// saliency_map.no_alias() += mat_weights_classifier * mat_img_avg_pool;

 // DEBUG: Make sure multiplication makes sense

 for(int i = 0; i < 49; ++i)
 {
     float mult_result = 0;
     for(int j = 0; j < weights_classifier.size(); ++j)
     {
         mult_result += mat_weights_classifier(j) * mat_img_avg_pool(1024*i + j);
     }
     std::cout << "Here is result of multiplication of weights and average pool: "
               << mult_result << std::endl;
 }


// std::cout << "Here is mat_weights_classifier * mat_img_avg_pool:\n"
//           << saliency_map << std::endl;

//  std::cout << "Complete" << std::endl;

    // Reshape saliency map
    Eigen::Map<Eigen::MatrixXf> saliency_map_reshape(saliency_map.data(), 7, 7);
    Eigen::MatrixXf saliency_matrix(7,7);
    saliency_matrix = saliency_map_reshape;
//    Eigen::MatrixXf saliency_matrix_transpose(7,7);
//    saliency_matrix_transpose = saliency_matrix.transpose();

    // Map to openCV
    cv::Mat_<float> saliency_matrix_opencv = cv::Mat_<float>::zeros(7,7);
//    cv::eigen2cv(saliency_matrix_transpose, saliency_matrix_opencv);
    cv::eigen2cv(saliency_matrix, saliency_matrix_opencv);

//    transpose(saliency_matrix_opencv,saliency_matrix_opencv);
    cv::Size scaleUp(1024,1024);
      cv::resize(saliency_matrix_opencv, saliency_matrix_opencv, scaleUp);
      std::cout << "scaled size: " << saliency_matrix_opencv.size() << std::endl;

      double min, max;
      cv::minMaxLoc(saliency_matrix_opencv, &min, &max);
      std::cout << "Min, max (before normalizing): " << min << ", " << max << std::endl;

//      saliency_matrix_opencv.convertTo(saliency_matrix_opencv, CV_32FC3);
      saliency_matrix_opencv -= min;
      cv::minMaxLoc(saliency_matrix_opencv, &min, &max);
      saliency_matrix_opencv /= max;
      saliency_matrix_opencv *= 255;

      cv::minMaxLoc(saliency_matrix_opencv, &min, &max);
      std::cout << "Min, max (after normalizing): " << min << ", " << max << std::endl;

      cv::Mat cm_saliency_matrix_opencv;
      // Apply the colormap:
      saliency_matrix_opencv.convertTo(cm_saliency_matrix_opencv, CV_8UC1);
      cv::applyColorMap(cm_saliency_matrix_opencv, cm_saliency_matrix_opencv, cv::COLORMAP_JET);

      cv::Scalar intensity = cm_saliency_matrix_opencv.at<uchar>(10,10);
      std::cout << intensity.val[0] << std::endl;

      // CV_8UC3
//      saliency_matrix_opencv.convertTo(saliency_matrix_opencv, CV_8U);
      cv::Mat cm_saliency_matrix_opencv_color;
//      cv::cvtColor(cm_saliency_matrix_opencv,cm_saliency_matrix_opencv_color,CV_GRAY2BGR);

      image = cv::imread(FLAGS_file);

      cv::Mat result;

      result = cm_saliency_matrix_opencv * 0.3 + image * 0.5;

      cv::imwrite("saliency_heatmap.jpg", result);

      // convert to float, normalize to mean 128
//      image.convertTo(image, CV_32FC3, 1.0, -128);
//      std::cout << "value range: ("
//                << *std::min_element((float *)image.datastart,
//                                     (float *)image.dataend)
//                << ", "
//                << *std::max_element((float *)image.datastart,
//                                     (float *)image.dataend)
//                << ")" << std::endl;

}

}   // namespace caffe2

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}

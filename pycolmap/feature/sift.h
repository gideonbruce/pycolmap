#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"

#include "pycolmap/helpers.h"
#include "pycolmap/utils.h"

#include <memory>

#include <Eigen/Core>
#include <FreeImage.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#define kdim 4

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

template <typename dtype>
using pyimage_t =
    Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    descriptors_t;
typedef Eigen::Matrix<float, Eigen::Dynamic, kdim, Eigen::RowMajor> keypoints_t;
typedef std::tuple<keypoints_t, descriptors_t> sift_output_t;

static std::map<int, std::unique_ptr<std::mutex>> sift_gpu_mutexes;

class Sift {
 public:
  Sift(SiftExtractionOptions options, Device device)
      : options_(std::move(options)), use_gpu_(IsGPU(device)) {
    VerifyGPUParams(use_gpu_);
    options_.use_gpu = use_gpu_;
    extractor_ = CreateSiftFeatureExtractor(options_);
    THROW_CHECK(extractor_ != nullptr);
  }

  sift_output_t Extract(const Eigen::Ref<const pyimage_t<uint8_t>>& image) {
    return ExtractInternal(image, Eigen::Ref<const pyimage_t<uint8_t>>());
  }

  sift_output_t Extract(const Eigen::Ref<const pyimage_t<float>>& image) {
    const pyimage_t<uint8_t> image_u8 = (image * 255.0f).cast<uint8_t>();
    return ExtractInternal(image_u8, Eigen::Ref<const pyimage_t<uint8_t>>());
  }

  sift_output_t Extract(const Eigen::Ref<const pyimage_t<uint8_t>>& image, 
                       const Eigen::Ref<const pyimage_t<uint8_t>>& mask) {
    return ExtractInternal(image, mask);
  }

  sift_output_t Extract(const Eigen::Ref<const pyimage_t<float>>& image, 
                       const Eigen::Ref<const pyimage_t<uint8_t>>& mask) {
    const pyimage_t<uint8_t> image_u8 = (image * 255.0f).cast<uint8_t>();
    return ExtractInternal(image_u8, mask);
  }

  const SiftExtractionOptions& Options() const { return options_; };

  Device GetDevice() const { return (use_gpu_) ? Device::CUDA : Device::CPU; };

 private:
  sift_output_t ExtractInternal(const Eigen::Ref<const pyimage_t<uint8_t>>& image,
                               const Eigen::Ref<const pyimage_t<uint8_t>>& mask) {
    THROW_CHECK_LE(image.rows(), options_.max_image_size);
    THROW_CHECK_LE(image.cols(), options_.max_image_size);

    // Validate mask dimensions if provided
    const bool has_mask = mask.size() > 0;
    if (has_mask) {
      THROW_CHECK_EQ(image.rows(), mask.rows());
      THROW_CHECK_EQ(image.cols(), mask.cols());
    }

    const unsigned int bpp = 8;  // Grey.
    const unsigned int width = image.cols();
    const unsigned int scan_width = (bpp / 8) * width;
    pyimage_t<uint8_t> image_copy = image;
    
    // Apply mask to image if provided
    if (has_mask) {
      for (int y = 0; y < image_copy.rows(); ++y) {
        for (int x = 0; x < image_copy.cols(); ++x) {
          if (mask(y, x) == 0) {
            image_copy(y, x) = 0;  // Set masked pixels to black
          }
        }
      }
    }

    FIBITMAP* bitmap_raw = FreeImage_ConvertFromRawBitsEx(
        /*copySource=*/false,
        static_cast<unsigned char*>(image_copy.data()),
        FIT_BITMAP,
        width,
        image.rows(),
        scan_width,
        bpp,
        FI_RGBA_RED_MASK,
        FI_RGBA_GREEN_MASK,
        FI_RGBA_BLUE_MASK,
        /*topdown=*/true);
    const Bitmap bitmap(bitmap_raw);

    FeatureKeypoints keypoints_;
    FeatureDescriptors descriptors_;
    THROW_CHECK(extractor_->Extract(bitmap, &keypoints_, &descriptors_));
    
    const size_t num_features = keypoints_.size();

    keypoints_t keypoints(num_features, kdim);
    for (size_t i = 0; i < num_features; ++i) {
      keypoints(i, 0) = keypoints_[i].x;
      keypoints(i, 1) = keypoints_[i].y;
      keypoints(i, 2) = keypoints_[i].ComputeScale();
      keypoints(i, 3) = keypoints_[i].ComputeOrientation();
    }

    descriptors_t descriptors = descriptors_.cast<float>();
    descriptors /= 512.0f;

    return std::make_tuple(keypoints, descriptors);
  }

  std::unique_ptr<FeatureExtractor> extractor_;
  SiftExtractionOptions options_;
  bool use_gpu_ = false;
};

void BindSift(py::module& m) {
  // For backwards consistency
  py::dict sift_options;
  sift_options["peak_threshold"] = 0.01;
  sift_options["first_octave"] = 0;
  sift_options["max_image_size"] = 7000;

  py::class_<Sift>(m, "Sift")
      .def(py::init<SiftExtractionOptions, Device>(),
           "options"_a = sift_options,
           "device"_a = Device::AUTO)
      
      // Extract with mask (uint8 image, uint8 mask)
      .def("extract",
           py::overload_cast<const Eigen::Ref<const pyimage_t<uint8_t>>&,
                           const Eigen::Ref<const pyimage_t<uint8_t>>&>(
               &Sift::Extract),
           "image"_a.noconvert(), "mask"_a.noconvert(),
           "Extract SIFT features from uint8 image with uint8 mask")
      
      // Extract with mask (float image, uint8 mask)
      .def("extract",
           py::overload_cast<const Eigen::Ref<const pyimage_t<float>>&,
                           const Eigen::Ref<const pyimage_t<uint8_t>>&>(
               &Sift::Extract),
           "image"_a.noconvert(), "mask"_a.noconvert(),
           "Extract SIFT features from float image with uint8 mask")
      
      // Extract without mask (uint8 image)
      .def("extract",
           py::overload_cast<const Eigen::Ref<const pyimage_t<uint8_t>>&>(
               &Sift::Extract),
           "image"_a.noconvert(),
           "Extract SIFT features from uint8 image")
      
      // Extract without mask (float image)
      .def("extract",
           py::overload_cast<const Eigen::Ref<const pyimage_t<float>>&>(
               &Sift::Extract),
           "image"_a.noconvert(),
           "Extract SIFT features from float image")
      
      .def_property_readonly("options", &Sift::Options)
      .def_property_readonly("device", &Sift::GetDevice);
}
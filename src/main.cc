#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <iostream>

#include "ORBextractor.h"
#include "NDArrayConverter.h"

namespace py = pybind11;

namespace pybind11
{
    namespace detail
    {

        template <>
        struct type_caster<cv::KeyPoint>
        {
        public:
            PYBIND11_TYPE_CASTER(cv::KeyPoint, _("cv2.KeyPoint"));

            bool load(handle src, bool)
            {
                py::tuple pt = reinterpret_borrow<py::tuple>(src.attr("pt"));
                auto x = pt[0].cast<float>();
                auto y = pt[1].cast<float>();
                auto size = src.attr("size").cast<float>();
                auto angle = src.attr("angle").cast<float>();
                auto response = src.attr("response").cast<float>();
                auto octave = src.attr("octave").cast<int>();
                auto class_id = src.attr("class_id").cast<int>();
                // (float x, float y, float _size, float _angle, float _response, int _octave, int _class_id)
                value = cv::KeyPoint(x, y, size, angle, response, octave, class_id);

                return true;
            }

            static handle cast(const cv::KeyPoint &kp, return_value_policy, handle defval)
            {
                auto classKP = py::module::import("cv2").attr("KeyPoint");
                auto cvKP = classKP(kp.pt.x, kp.pt.y, kp.size, kp.angle, kp.response, kp.octave, kp.class_id);
                return cvKP.release();
            }
        };

    }
}

class PyORBExtractor
{
private:
    ORB_SLAM3::ORBextractor *extractor;

public:
    PyORBExtractor(int nfeatures, float scaleFactor, int nlevels,
                   int iniThFAST, int minThFAST, int interpolation, bool angle);
    ~PyORBExtractor();

    py::tuple detectAndCompute(cv::Mat image, cv::Mat mask, std::vector<int> lappingArea);
};

PyORBExtractor::PyORBExtractor(int nfeatures, float scaleFactor, int nlevels,
                               int iniThFAST, int minThFAST, int interpolation, bool angle)
{
    extractor = new ORB_SLAM3::ORBextractor(nfeatures, scaleFactor, nlevels,
                                            iniThFAST, minThFAST, interpolation, angle);
}

PyORBExtractor::~PyORBExtractor()
{
    delete extractor;
}

py::tuple PyORBExtractor::detectAndCompute(cv::Mat image, cv::Mat mask, std::vector<int> lappingArea)
{
    // std::cout << "detectAndCompute, type:" << image.type() << std::endl;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    auto result = extractor->operator()(image, mask, keypoints, descriptors, lappingArea);

    return py::make_tuple(keypoints, descriptors);
}

PYBIND11_MODULE(orb_slam3, m)
{
    NDArrayConverter::init_numpy();

    m.doc() = "ORB_SLAM3";

    py::class_<PyORBExtractor>(m, "ORBExtractor")
        .def(py::init<int, float, int, int, int, int, bool>(),
             py::arg("nfeatures") = 1000, py::arg("scaleFactor") = 1.2,
             py::arg("nlevels") = 8, py::arg("iniThFAST") = 20,
             py::arg("minThFAST") = 7, py::arg("interpolation") = 1,
             py::arg("angle") = true)
        .def("detectAndCompute", &PyORBExtractor::detectAndCompute,
             py::arg("image"), py::arg("mask") = cv::Mat(),
             py::arg("lappingArea") = std::vector<int>({0, 0}));
}

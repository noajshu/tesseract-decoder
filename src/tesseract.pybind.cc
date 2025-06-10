#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stim.h"
#include "tesseract.h"

namespace py = pybind11;

struct CompiledDecoder {
  TesseractDecoder decoder;

  static CompiledDecoder from_dem(const py::object &dem_obj) {
    std::string dem_str = py::str(dem_obj);
    stim::DetectorErrorModel dem(dem_str.c_str());
    TesseractConfig cfg{dem};
    return CompiledDecoder{TesseractDecoder(cfg)};
  }

  uint64_t decode(const py::iterable &det_events) {
    std::vector<uint64_t> dets;
    for (const auto &d : det_events) {
      dets.push_back(py::cast<uint64_t>(d));
    }
    return decoder.decode(dets);
  }
};

PYBIND11_MODULE(tesseract_decoder, m) {
  py::class_<CompiledDecoder>(m, "CompiledDecoder")
      .def_static("from_dem", &CompiledDecoder::from_dem, py::arg("dem"))
      .def("decode", &CompiledDecoder::decode, py::arg("detections"));

  m.def("compile_decoder_for_dem", &CompiledDecoder::from_dem,
        py::arg("dem"));
}

# In Progress Tasks

Nhiệm vụ đang thực hiện:

## 🎯 P0.1: Xây Dựng Bộ Benchmark Recognition Nội Bộ & Đánh Giá ArcFace Baseline
- [x] Cập nhật tài liệu chiến lược `docs/00_MASTER_PLAN.md` và tiêu chuẩn `docs/03_BENCHMARK.md` theo định hướng mới.
- [ ] Thiết lập cây thư mục bộ benchmark: `benchmarks/gallery/`, `benchmarks/probe/`, `benchmarks/pairs/`, `benchmarks/results/`.
- [ ] Xây dựng `requirements.txt` và thiết lập môi trường Python.
- [ ] Xây dựng module căn chỉnh khuôn mặt 5 landmarks `src/alignment/aligner.py` (Umeyama Affine Transform chuẩn 112x112).
- [ ] Xây dựng interface `BaseFaceRecognizer` và ArcFace ONNX wrapper `src/recognition/arcface.py` (512D L2-normalized).
- [ ] Xây dựng bộ công cụ benchmark: `generate_pairs.py`, `metrics.py`, `run_benchmark.py`.

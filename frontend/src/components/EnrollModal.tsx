import React, { useState, useRef } from 'react';
import { X, Camera, Upload, CheckCircle2, AlertTriangle, RefreshCw, UserPlus } from 'lucide-react';
import { createPerson, enrollFace } from '../services/api';

interface EnrollModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export const EnrollModal: React.FC<EnrollModalProps> = ({ isOpen, onClose, onSuccess }) => {
  const [personId, setPersonId] = useState('');
  const [name, setName] = useState('');
  const [department, setDepartment] = useState('');
  const [role, setRole] = useState('Nhân viên');
  const [captureMode, setCaptureMode] = useState<'webcam' | 'file'>('webcam');

  const [imageBlob, setImageBlob] = useState<Blob | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [countdown, setCountdown] = useState<number | null>(null);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // Webcam stream refs for modal
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);

  // Start webcam when modal opens or switches to webcam mode
  const startModalCamera = async () => {
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      let stream: MediaStream | null = null;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        });
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch(() => {});
      }
      setIsCameraActive(true);
    } catch (err: any) {
      const name = err?.name || '';
      if (name === 'NotReadableError' || name === 'TrackStartError') {
        setErrorMsg('Camera đang bị ứng dụng khác sử dụng (ví dụ scripts/webcam_demo.py). Vui lòng đóng ứng dụng đó!');
      } else if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        setErrorMsg('Trình duyệt chưa được cấp quyền Camera. Hãy bấm icon Ổ khóa trên thanh địa chỉ URL để Cho phép.');
      } else {
        setErrorMsg('Không thể mở camera. Bạn có thể chọn cách "Tải file ảnh".');
      }
    }
  };

  const stopModalCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
  };

  const handleCapturePhoto = () => {
    setCountdown(3);
    let count = 3;
    const interval = setInterval(() => {
      count -= 1;
      if (count > 0) {
        setCountdown(count);
      } else {
        clearInterval(interval);
        setCountdown(null);
        // Snapshot
        const video = videoRef.current;
        if (video) {
          const canvas = document.createElement('canvas');
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(
              (blob) => {
                if (blob) {
                  setImageBlob(blob);
                  setImagePreview(canvas.toDataURL('image/jpeg'));
                  stopModalCamera();
                }
              },
              'image/jpeg',
              0.95
            );
          }
        }
      }
    }, 1000);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageBlob(file);
      const reader = new FileReader();
      reader.onload = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleResetImage = () => {
    setImageBlob(null);
    setImagePreview(null);
    if (captureMode === 'webcam') {
      startModalCamera();
    }
  };

  const handleModalClose = () => {
    stopModalCamera();
    setImageBlob(null);
    setImagePreview(null);
    setErrorMsg(null);
    setSuccessMsg(null);
    onClose();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!personId.trim() || !name.trim()) {
      setErrorMsg('Vui lòng nhập Mã định danh và Họ tên!');
      return;
    }
    if (!imageBlob) {
      setErrorMsg('Vui lòng chụp ảnh hoặc tải ảnh khuôn mặt lên!');
      return;
    }

    setIsSubmitting(true);
    setErrorMsg(null);
    setSuccessMsg(null);

    try {
      // 1. Tạo person trong DB (bỏ qua nếu đã tồn tại)
      try {
        await createPerson({
          person_id: personId.trim(),
          name: name.trim(),
          department: department.trim() || undefined,
          role: role.trim() || undefined,
        });
      } catch (err: any) {
        // Nếu person đã tồn tại, tiếp tục enroll face
        if (!err.message?.includes('already exists') && !err.message?.includes('đã tồn tại')) {
          throw err;
        }
      }

      // 2. Enroll face embedding
      await enrollFace(personId.trim(), imageBlob);

      setSuccessMsg(`Đăng ký khuôn mặt thành công cho ${name}!`);
      setTimeout(() => {
        onSuccess();
        handleModalClose();
      }, 1500);
    } catch (err: any) {
      setErrorMsg(err?.message || 'Có lỗi xảy ra khi lưu khuôn mặt');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-slate-800 w-full max-w-xl rounded-2xl shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="p-2 rounded-xl bg-cyan-950/80 text-cyan-400 border border-cyan-800/40">
              <UserPlus className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-white text-base">Đăng Ký Nhân Sự Mới</h3>
              <p className="text-xs text-slate-400">Trích xuất vector 512D ArcFace vào CSDL</p>
            </div>
          </div>
          <button
            onClick={handleModalClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {/* Alerts */}
          {errorMsg && (
            <div className="p-3 rounded-xl bg-rose-950/80 border border-rose-800 text-rose-300 text-xs flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
              <span>{errorMsg}</span>
            </div>
          )}
          {successMsg && (
            <div className="p-3 rounded-xl bg-emerald-950/80 border border-emerald-800 text-emerald-300 text-xs flex items-center space-x-2">
              <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
              <span>{successMsg}</span>
            </div>
          )}

          {/* Form Fields */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Mã định danh (ID) *
              </label>
              <input
                type="text"
                required
                value={personId}
                onChange={(e) => setPersonId(e.target.value)}
                placeholder="VD: NV005"
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500 transition"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Họ và tên *
              </label>
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="VD: Nguyễn Văn A"
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500 transition"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Phòng ban
              </label>
              <input
                type="text"
                value={department}
                onChange={(e) => setDepartment(e.target.value)}
                placeholder="VD: Kỹ thuật / Nhân sự"
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500 transition"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Chức vụ
              </label>
              <input
                type="text"
                value={role}
                onChange={(e) => setRole(e.target.value)}
                placeholder="VD: Kỹ sư / Trưởng phòng"
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500 transition"
              />
            </div>
          </div>

          {/* Photo Capture Section */}
          <div className="pt-2">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold text-slate-300">
                Ảnh khuôn mặt đăng ký
              </label>
              <div className="flex items-center space-x-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
                <button
                  type="button"
                  onClick={() => {
                    setCaptureMode('webcam');
                    startModalCamera();
                  }}
                  className={`px-2.5 py-0.5 rounded-md transition ${
                    captureMode === 'webcam' ? 'bg-cyan-600 text-white' : 'text-slate-400'
                  }`}
                >
                  Chụp Webcam
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setCaptureMode('file');
                    stopModalCamera();
                  }}
                  className={`px-2.5 py-0.5 rounded-md transition ${
                    captureMode === 'file' ? 'bg-cyan-600 text-white' : 'text-slate-400'
                  }`}
                >
                  Tải file ảnh
                </button>
              </div>
            </div>

            {/* Viewport for photo or upload */}
            <div className="aspect-video w-full rounded-xl bg-slate-950 border border-slate-800 relative overflow-hidden flex items-center justify-center">
              {imagePreview ? (
                // Image preview with Retake button
                <div className="relative w-full h-full">
                  <img
                    src={imagePreview}
                    alt="Preview"
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-slate-950/80 via-transparent flex items-end justify-center p-3">
                    <button
                      type="button"
                      onClick={handleResetImage}
                      className="px-3 py-1.5 bg-slate-800/90 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-medium border border-slate-600 flex items-center space-x-1.5 transition"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                      <span>Chụp / Chọn lại</span>
                    </button>
                  </div>
                </div>
              ) : captureMode === 'webcam' ? (
                // Webcam Capture view
                <div className="relative w-full h-full flex items-center justify-center">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover transform -scale-x-100"
                  />
                  {countdown !== null && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-xs">
                      <span className="text-6xl font-bold font-mono text-cyan-400 animate-ping">
                        {countdown}
                      </span>
                    </div>
                  )}
                  {isCameraActive && countdown === null && (
                    <div className="absolute bottom-3 inset-x-0 flex justify-center">
                      <button
                        type="button"
                        onClick={handleCapturePhoto}
                        className="px-4 py-1.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold flex items-center space-x-1.5 shadow-lg shadow-cyan-500/30 transition active:scale-95"
                      >
                        <Camera className="w-4 h-4" />
                        <span>Chụp ảnh (3s)</span>
                      </button>
                    </div>
                  )}
                  {!isCameraActive && (
                    <button
                      type="button"
                      onClick={startModalCamera}
                      className="px-4 py-2 bg-slate-800 text-slate-300 rounded-xl text-xs flex items-center space-x-2 border border-slate-700 hover:bg-slate-700 transition"
                    >
                      <Camera className="w-4 h-4 text-cyan-400" />
                      <span>Kích hoạt Camera</span>
                    </button>
                  )}
                </div>
              ) : (
                // File upload dropzone
                <label className="flex flex-col items-center justify-center w-full h-full cursor-pointer hover:bg-slate-900/50 transition">
                  <Upload className="w-8 h-8 text-slate-500 mb-2" />
                  <span className="text-xs text-slate-300 font-medium">
                    Nhấp để chọn ảnh chân dung
                  </span>
                  <span className="text-[11px] text-slate-500 mt-0.5">JPG, PNG chất lượng rõ nét</span>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </label>
              )}
            </div>
          </div>

          {/* Action buttons */}
          <div className="pt-3 border-t border-slate-800 flex items-center justify-end space-x-3">
            <button
              type="button"
              onClick={handleModalClose}
              className="px-4 py-2 text-xs font-medium text-slate-400 hover:text-white transition"
            >
              Hủy
            </button>
            <button
              type="submit"
              disabled={isSubmitting || !imageBlob}
              className={`px-5 py-2 rounded-xl text-xs font-semibold bg-cyan-600 hover:bg-cyan-500 text-white shadow-md shadow-cyan-600/30 flex items-center space-x-2 transition ${
                isSubmitting || !imageBlob ? 'opacity-50 cursor-not-allowed' : 'active:scale-95'
              }`}
            >
              {isSubmitting && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
              <span>{isSubmitting ? 'Đang trích xuất vector...' : 'Lưu & Đăng ký'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

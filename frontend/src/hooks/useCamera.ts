import { useRef, useState, useCallback, useEffect } from 'react';

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isActive, setIsActive] = useState<boolean>(false);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsActive(false);
    setIsStreaming(false);
  }, []);

  const parseCameraError = (err: any): string => {
    const name = err?.name || '';
    if (name === 'NotReadableError' || name === 'TrackStartError') {
      return 'Camera đang bị ứng dụng khác chiếm dụng (ví dụ: scripts/webcam_demo.py hoặc phần mềm khác). Vui lòng đóng ứng dụng đang dùng camera và bấm lại "Bật Camera"!';
    }
    if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
      return 'Trình duyệt bị chặn quyền Camera. Vui lòng bấm vào biểu tượng Ổ Khóa (hoặc icon Camera) cạnh thanh địa chỉ URL -> Chọn "Cho phép (Allow)" rồi tải lại trang.';
    }
    if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
      return 'Không tìm thấy thiết bị Camera nào được cắm vào máy tính.';
    }
    if (name === 'OverconstrainedError') {
      return 'Camera không hỗ trợ độ phân giải yêu cầu. Đang thử chế độ cơ bản...';
    }
    return err?.message || 'Không thể truy cập camera. Vui lòng kiểm tra quyền và thiết bị!';
  };

  const startCamera = useCallback(async (deviceId?: string) => {
    stopCamera();
    setError(null);

    let stream: MediaStream | null = null;

    // 1. First attempt with ideal HD resolution
    try {
      const constraints: MediaStreamConstraints = {
        video: deviceId
          ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
          : { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      };
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (primaryErr: any) {
      // 2. Fallback attempt: if overconstrained or specific mode failed, try basic { video: true }
      if (primaryErr?.name === 'OverconstrainedError' || primaryErr?.name === 'ConstraintNotSatisfiedError') {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        } catch (fallbackErr: any) {
          setError(parseCameraError(fallbackErr));
          setIsActive(false);
          setIsStreaming(false);
          return;
        }
      } else {
        setError(parseCameraError(primaryErr));
        setIsActive(false);
        setIsStreaming(false);
        return;
      }
    }

    if (!stream) return;

    try {
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().catch(() => {});
          setIsStreaming(true);
        };
      }
      setIsActive(true);
    } catch (err: any) {
      setError(parseCameraError(err));
      setIsActive(false);
      setIsStreaming(false);
    }
  }, [stopCamera]);

  const captureFrameBlob = useCallback(async (quality: number = 0.92): Promise<Blob | null> => {
    const video = videoRef.current;
    if (!video || !isStreaming || video.videoWidth === 0 || video.videoHeight === 0) {
      return null;
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    return new Promise((resolve) => {
      canvas.toBlob(
        (blob) => {
          resolve(blob);
        },
        'image/jpeg',
        quality
      );
    });
  }, [isStreaming]);

  const captureFrameDataUrl = useCallback((): string | null => {
    const video = videoRef.current;
    if (!video || !isStreaming || video.videoWidth === 0 || video.videoHeight === 0) {
      return null;
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.9);
  }, [isStreaming]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return {
    videoRef,
    isActive,
    isStreaming,
    error,
    startCamera,
    stopCamera,
    captureFrameBlob,
    captureFrameDataUrl,
  };
}

import React from 'react';
import { Camera, CameraOff, RefreshCw, Zap, AlertTriangle, ShieldCheck, UserX, EyeOff, Radio } from 'lucide-react';
import { RecognitionResult } from '../types';

interface CameraFeedProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  isActive: boolean;
  isStreaming: boolean;
  error: string | null;
  onStartCamera: () => void;
  onStopCamera: () => void;
  autoScan: boolean;
  onToggleAutoScan: () => void;
  isScanning: boolean;
  onManualScan: () => void;
  lastResult: RecognitionResult | null;
  inferenceTimeMs: number | null;
}

export const CameraFeed: React.FC<CameraFeedProps> = ({
  videoRef,
  isActive,
  isStreaming,
  error,
  onStartCamera,
  onStopCamera,
  autoScan,
  onToggleAutoScan,
  isScanning,
  onManualScan,
  lastResult,
  inferenceTimeMs,
}) => {
  const getStatusBadge = () => {
    if (!lastResult) {
      return (
        <div className="flex items-center space-x-2 text-slate-400 text-xs">
          <span className="w-2 h-2 rounded-full bg-slate-500" />
          <span>Sẵn sàng quét khuôn mặt</span>
        </div>
      );
    }

    switch (lastResult.status) {
      case 'MATCHED':
        return (
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-emerald-950/80 border border-emerald-500/50 text-emerald-300">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <div className="text-xs">
              <span className="font-bold text-white text-sm">{lastResult.person_id}</span>
              <span className="ml-2 opacity-90">
                (Khớp {Math.round((lastResult.similarity ?? 0) * 100)}% - Margin: {(lastResult.margin ?? 0).toFixed(2)})
              </span>
            </div>
          </div>
        );
      case 'UNKNOWN':
        return (
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-amber-950/80 border border-amber-500/50 text-amber-300">
            <UserX className="w-4 h-4 text-amber-400" />
            <span className="text-xs font-semibold">
              Người lạ / Chưa đăng ký ({Math.round((lastResult.similarity ?? 0) * 100)}%)
            </span>
          </div>
        );
      case 'MASK_DETECTED':
        return (
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-rose-950/80 border border-rose-500/60 text-rose-300 animate-pulse-fast">
            <AlertTriangle className="w-4 h-4 text-rose-400" />
            <span className="text-xs font-bold">
              Phát hiện khẩu trang! Vui lòng tháo khẩu trang để nhận diện
            </span>
          </div>
        );
      case 'OCCLUDED_FACE':
        return (
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-rose-950/80 border border-rose-500/60 text-rose-300 animate-pulse-fast">
            <EyeOff className="w-4 h-4 text-rose-400" />
            <span className="text-xs font-bold">
              Khuôn mặt bị che khuất! Vui lòng bỏ tay / vật cản
            </span>
          </div>
        );
      case 'NO_FACE':
        return (
          <div className="flex items-center space-x-2 text-slate-400 text-xs">
            <Radio className="w-3.5 h-3.5 text-slate-500 animate-spin" />
            <span>Đang tìm kiếm khuôn mặt trong khung hình...</span>
          </div>
        );
      default:
        return (
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-300 text-xs">
            <span>Trạng thái: {lastResult.status}</span>
          </div>
        );
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl flex flex-col justify-between">
      {/* Top Header of the Camera Card */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <div className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
          <h2 className="text-sm font-semibold tracking-wide text-slate-200">
            TRỰC TIẾP CAMERA (LIVE AI FEED)
          </h2>
        </div>

        {/* Latency badge */}
        {inferenceTimeMs !== null && (
          <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-slate-800 border border-slate-700 text-cyan-400">
            AI: {inferenceTimeMs}ms
          </span>
        )}
      </div>

      {/* Video Viewport Container */}
      <div className="relative aspect-video w-full rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center group shadow-inner">
        {isActive ? (
          <>
            {/* The video element with selfie mirror effect */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover transform -scale-x-100"
            />

            {/* Cyberpunk HUD Overlay */}
            {isStreaming && (
              <div className="absolute inset-0 pointer-events-none p-4 flex flex-col justify-between">
                {/* 4 Corner Brackets */}
                <div className="flex justify-between items-start">
                  <div className="w-8 h-8 border-t-2 border-l-2 border-cyan-400/80 rounded-tl" />
                  <div className="w-8 h-8 border-t-2 border-r-2 border-cyan-400/80 rounded-tr" />
                </div>

                {/* Animated Horizontal Scanline */}
                {isScanning && (
                  <div className="absolute left-4 right-4 h-0.5 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_12px_rgba(6,182,212,0.8)] animate-scan" />
                )}

                {/* Center Targeting Box */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-48 h-56 sm:w-56 sm:h-64 border border-dashed border-cyan-500/40 rounded-3xl relative flex items-center justify-center">
                    <div className="w-4 h-4 border-t border-l border-cyan-400 absolute top-2 left-2" />
                    <div className="w-4 h-4 border-t border-r border-cyan-400 absolute top-2 right-2" />
                    <div className="w-4 h-4 border-b border-l border-cyan-400 absolute bottom-2 left-2" />
                    <div className="w-4 h-4 border-b border-r border-cyan-400 absolute bottom-2 right-2" />
                    {/* Small center reticle dot */}
                    <div className="w-1.5 h-1.5 rounded-full bg-cyan-400/60" />
                  </div>
                </div>

                {/* Bottom Corner Brackets */}
                <div className="flex justify-between items-end">
                  <div className="w-8 h-8 border-b-2 border-l-2 border-cyan-400/80 rounded-bl" />
                  <div className="w-8 h-8 border-b-2 border-r-2 border-cyan-400/80 rounded-br" />
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="flex flex-col items-center justify-center text-slate-500 space-y-3 p-6 text-center">
            <div className="w-16 h-16 rounded-full bg-slate-900 flex items-center justify-center border border-slate-800">
              <CameraOff className="w-8 h-8 text-slate-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-400">Camera đang tắt</p>
              <p className="text-xs text-slate-600 mt-1">
                Bấm nút &quot;Bật Camera&quot; bên dưới để bắt đầu nhận diện
              </p>
            </div>
          </div>
        )}

        {/* Error notification banner */}
        {error && (
          <div className="absolute bottom-4 left-4 right-4 bg-rose-950/90 border border-rose-800 text-rose-300 px-4 py-2 rounded-lg text-xs flex items-center space-x-2 backdrop-blur-md">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Recognition Result Banner */}
      <div className="mt-3 min-h-[44px] flex items-center justify-center">
        {getStatusBadge()}
      </div>

      {/* Control Buttons Bar */}
      <div className="mt-3 pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-3">
        {/* Left: Camera On/Off */}
        <div>
          {isActive ? (
            <button
              onClick={onStopCamera}
              className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl text-xs font-medium bg-slate-800 hover:bg-slate-700 text-rose-400 border border-slate-700 transition active:scale-95"
            >
              <CameraOff className="w-3.5 h-3.5" />
              <span>Tắt Camera</span>
            </button>
          ) : (
            <button
              onClick={onStartCamera}
              className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl text-xs font-medium bg-cyan-600 hover:bg-cyan-500 text-white shadow-md shadow-cyan-600/30 transition active:scale-95"
            >
              <Camera className="w-3.5 h-3.5" />
              <span>Bật Camera</span>
            </button>
          )}
        </div>

        {/* Right: Auto Scan Toggle & Manual Scan Button */}
        <div className="flex items-center space-x-2">
          {/* Auto Scan Toggle Switch */}
          <button
            disabled={!isActive}
            onClick={onToggleAutoScan}
            className={`flex items-center space-x-2 px-3 py-2 rounded-xl text-xs font-medium border transition ${
              autoScan
                ? 'bg-emerald-950/60 border-emerald-700 text-emerald-300'
                : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-300'
            } ${!isActive ? 'opacity-50 cursor-not-allowed' : 'active:scale-95'}`}
            title="Tự động chụp và gửi đến FastAPI mỗi 1.5 giây"
          >
            <Zap className={`w-3.5 h-3.5 ${autoScan ? 'text-emerald-400 fill-emerald-400' : ''}`} />
            <span>Tự động quét: {autoScan ? 'BẬT' : 'TẮT'}</span>
          </button>

          {/* Manual Snapshot Button */}
          <button
            disabled={!isActive || isScanning}
            onClick={onManualScan}
            className={`flex items-center space-x-1.5 px-4 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white shadow-md shadow-blue-500/20 transition ${
              !isActive || isScanning ? 'opacity-50 cursor-not-allowed' : 'active:scale-95'
            }`}
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isScanning ? 'animate-spin' : ''}`} />
            <span>{isScanning ? 'Đang nhận diện...' : 'Quét ngay'}</span>
          </button>
        </div>
      </div>
    </div>
  );
};

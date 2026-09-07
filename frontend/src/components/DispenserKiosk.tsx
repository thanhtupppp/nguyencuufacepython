import React, { useState, useEffect, useRef } from 'react';
import {
  Camera,
  CameraOff,
  Clock,
  CheckCircle2,
  AlertTriangle,
  Volume2,
  VolumeX,
  Maximize2,
  Minimize2,
  ShieldCheck,
  UserPlus,
  RefreshCw,
  Sparkles,
  Settings,
  Scan,
} from 'lucide-react';
import { DispenseResult } from '../types';
import { checkPresence } from '../services/api';
import { soundEffects } from '../utils/audio';
import { FaceViewfinder, ViewfinderStyle } from './FaceViewfinder';

interface DispenserKioskProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  isActive: boolean;
  isStreaming: boolean;
  cameraError: string | null;
  onStartCamera: () => void;
  onStopCamera: () => void;
  onRequestPaper: (cooldownMinutes: number) => Promise<DispenseResult | null>;
  isProcessing: boolean;
  cooldownMinutes: number;
  onChangeCooldown: (minutes: number) => void;
  onOpenSettings: () => void;
  viewfinderStyle?: ViewfinderStyle;
  voiceEnabled?: boolean;
  voiceVolume?: number;
  voiceRate?: number;
  touchlessEnabled?: boolean;
  touchlessDelay?: number;
  welcomeVoiceEnabled?: boolean;
}

export const DispenserKiosk: React.FC<DispenserKioskProps> = ({
  videoRef,
  isActive,
  isStreaming,
  cameraError,
  onStartCamera,
  onStopCamera,
  onRequestPaper,
  isProcessing,
  cooldownMinutes,
  onChangeCooldown,
  onOpenSettings,
  viewfinderStyle = 'hud',
  voiceEnabled = true,
  voiceVolume = 1.0,
  voiceRate = 1.0,
  touchlessEnabled = true,
  touchlessDelay = 1.5,
  welcomeVoiceEnabled = true,
}) => {
  const [result, setResult] = useState<DispenseResult | null>(null);
  const [soundEnabled, setSoundEnabled] = useState<boolean>(voiceEnabled);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [countdownRemaining, setCountdownRemaining] = useState<number>(0);
  const [autoDismissSeconds, setAutoDismissSeconds] = useState<number>(0);

  // Smart Presence & Touchless State
  const [isIdle, setIsIdle] = useState<boolean>(true);
  const [presenceDetected, setPresenceDetected] = useState<boolean>(false);
  const [touchlessCountdown, setTouchlessCountdown] = useState<number>(0);
  const [isDebouncing, setIsDebouncing] = useState<boolean>(false);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const isCheckingPresenceRef = useRef<boolean>(false);
  const lastPresenceTimeRef = useRef<number>(0);
  const hasGreetedRef = useRef<boolean>(false);
  const isProcessingRef = useRef<boolean>(isProcessing);
  const isDebouncingRef = useRef<boolean>(false);

  useEffect(() => {
    isProcessingRef.current = isProcessing;
  }, [isProcessing]);

  // Sync audio/voice config with soundEffects controller
  useEffect(() => {
    soundEffects.setSoundEnabled(soundEnabled);
    soundEffects.setVoiceConfig(soundEnabled && voiceEnabled, voiceVolume, voiceRate);
  }, [soundEnabled, voiceEnabled, voiceVolume, voiceRate]);

  // Offscreen fast frame capture (320x240 for <15ms SCRFD presence checking)
  const captureSmallFrame = (): Promise<Blob | null> => {
    if (!videoRef.current || videoRef.current.readyState < 2) return Promise.resolve(null);
    const video = videoRef.current;
    if (!offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement('canvas');
      offscreenCanvasRef.current.width = 320;
      offscreenCanvasRef.current.height = 240;
    }
    const canvas = offscreenCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return Promise.resolve(null);
    ctx.drawImage(video, 0, 0, 320, 240);
    return new Promise<Blob | null>((resolve) => {
      canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.65);
    });
  };

  // Trigger paper request
  const handleDispenseClick = async () => {
    if (isProcessing) return;
    setTouchlessCountdown(0);
    setResult(null);

    const res = await onRequestPaper(cooldownMinutes);
    if (!res) return;

    setResult(res);
    setAutoDismissSeconds(3);

    // Cooldown debounce (6s) so person can gather paper and step away without re-triggering
    setIsDebouncing(true);
    isDebouncingRef.current = true;
    setTimeout(() => {
      setIsDebouncing(false);
      isDebouncingRef.current = false;
      hasGreetedRef.current = false;
    }, 6000);

    if (res.granted) {
      if (soundEnabled) {
        soundEffects.announceGranted(res.name, res.is_new_user);
      }
    } else {
      if (soundEnabled) {
        if (res.status === 'COOLDOWN_BLOCKED') {
          soundEffects.announceBlocked(res.cooldown_remaining_seconds || 60);
        } else if (res.status === 'MASK_DETECTED') {
          soundEffects.announceMaskDetected();
        } else if (res.status === 'OCCLUSION_DETECTED') {
          soundEffects.announceOcclusion();
        } else if (res.status === 'NO_FACE_DETECTED') {
          soundEffects.announceNoFace();
        } else if (res.status === 'SPOOF_DETECTED') {
          soundEffects.announceSpoofDetected();
        } else {
          soundEffects.playBlocked();
        }
      }
      if (res.status === 'COOLDOWN_BLOCKED' && res.cooldown_remaining_seconds) {
        setCountdownRemaining(res.cooldown_remaining_seconds);
      }
    }
  };

  // Smart Presence Check Loop (Runs every 750ms when camera is active)
  useEffect(() => {
    if (!isActive || !isStreaming) {
      setIsIdle(true);
      setPresenceDetected(false);
      setTouchlessCountdown(0);
      return;
    }

    const interval = setInterval(async () => {
      if (
        isProcessingRef.current ||
        isDebouncingRef.current ||
        isCheckingPresenceRef.current ||
        result !== null ||
        countdownRemaining > 0
      ) {
        return;
      }

      try {
        isCheckingPresenceRef.current = true;
        const blob = await captureSmallFrame();
        if (!blob) return;

        const pres = await checkPresence(blob);
        if (pres.face_detected) {
          lastPresenceTimeRef.current = Date.now();
          setIsIdle(false);
          setPresenceDetected(true);

          // Welcome greeting when someone walks up
          if (welcomeVoiceEnabled && soundEnabled && !hasGreetedRef.current) {
            hasGreetedRef.current = true;
            soundEffects.announceWelcome();
          }

          // Trigger touchless countdown if enabled and not already running
          if (touchlessEnabled && touchlessCountdown <= 0 && !isProcessingRef.current) {
            setTouchlessCountdown(touchlessDelay);
          }
        } else {
          setPresenceDetected(false);
          // If person steps away, cancel countdown immediately
          setTouchlessCountdown(0);

          // Return to idle sleep mode after 3.5s of empty camera view
          if (Date.now() - lastPresenceTimeRef.current > 3500) {
            setIsIdle(true);
            hasGreetedRef.current = false;
          }
        }
      } catch {
        // Background polling errors ignored
      } finally {
        isCheckingPresenceRef.current = false;
      }
    }, 750);

    return () => clearInterval(interval);
  }, [
    isActive,
    isStreaming,
    result,
    countdownRemaining,
    touchlessEnabled,
    touchlessDelay,
    welcomeVoiceEnabled,
    soundEnabled,
    touchlessCountdown,
  ]);

  // Touchless Auto-Dispense live countdown
  useEffect(() => {
    if (!touchlessEnabled || touchlessCountdown <= 0 || isProcessing || isDebouncing) return;

    const timer = setInterval(() => {
      setTouchlessCountdown((prev) => {
        if (prev <= 0.15) {
          clearInterval(timer);
          handleDispenseClick();
          return 0;
        }
        return Number((prev - 0.1).toFixed(1));
      });
    }, 100);

    return () => clearInterval(timer);
  }, [touchlessEnabled, touchlessCountdown, isProcessing, isDebouncing]);

  // Cooldown timer live countdown
  useEffect(() => {
    if (countdownRemaining <= 0) return;
    const timer = setInterval(() => {
      setCountdownRemaining((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [countdownRemaining]);

  // Auto-dismiss result card
  useEffect(() => {
    if (autoDismissSeconds <= 0) return;
    const timer = setInterval(() => {
      setAutoDismissSeconds((prev) => {
        if (prev <= 1) {
          setResult(null);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [autoDismissSeconds]);

  // Fullscreen toggle
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {});
    }
  };

  const formatSeconds = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    if (m > 0) return `${m} phút ${s < 10 ? '0' : ''}${s} giây`;
    return `${s} giây`;
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full bg-slate-950 rounded-3xl border border-slate-800/80 shadow-2xl overflow-hidden flex flex-col items-center justify-between min-h-[600px] p-4 sm:p-6 select-none"
    >
      {/* Kiosk Top Controls Bar */}
      <div className="w-full flex items-center justify-between z-10 mb-2">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 text-white font-bold text-sm">
            🧻
          </div>
          <div>
            <h2 className="text-sm sm:text-base font-bold text-white tracking-wide flex items-center gap-1.5">
              KIOSK CẤP GIẤY VỆ SINH THÔNG MINH
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800 font-mono">
                AI Touchless
              </span>
            </h2>
            <p className="text-[11px] text-slate-400">Nhận diện khuôn mặt • Chống lạm dụng Cooldown 5 phút</p>
          </div>
        </div>

        {/* Right Tool Buttons */}
        <div className="flex items-center space-x-2">
          {/* Cooldown setting quick badge & Settings open */}
          <button
            onClick={onOpenSettings}
            className="flex items-center space-x-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-xl px-2.5 py-1.5 text-xs text-slate-300 transition"
            title="Nhấp để cấu hình chi tiết thời gian chống lạm dụng"
          >
            <Clock className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-[11px] text-slate-400">Chờ:</span>
            <span className="text-xs font-mono font-bold text-cyan-300 bg-cyan-950 px-1.5 py-0.2 rounded border border-cyan-800/80">
              {cooldownMinutes}p
            </span>
            <Settings className="w-3.5 h-3.5 text-slate-400 ml-0.5" />
          </button>

          {/* Sound Toggle */}
          <button
            onClick={() => setSoundEnabled(!soundEnabled)}
            className="p-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-300 hover:text-white transition"
            title={soundEnabled ? 'Tắt âm thanh' : 'Bật âm thanh'}
          >
            {soundEnabled ? <Volume2 className="w-4 h-4 text-cyan-400" /> : <VolumeX className="w-4 h-4 text-slate-500" />}
          </button>

          {/* Dedicated Settings Button */}
          <button
            onClick={onOpenSettings}
            className="p-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-300 hover:text-cyan-400 transition"
            title="Cài đặt thông số Kiosk & Chống lạm dụng"
          >
            <Settings className="w-4 h-4" />
          </button>

          {/* Fullscreen Toggle */}
          <button
            onClick={toggleFullscreen}
            className="p-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-300 hover:text-white transition"
            title="Toàn màn hình Kiosk"
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Main Viewport & Video Feed */}
      <div className="relative w-full max-w-2xl aspect-[4/3] sm:aspect-video rounded-2xl overflow-hidden bg-slate-900 border-2 border-slate-800 flex items-center justify-center shadow-inner group my-2">
        {isActive ? (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover transform -scale-x-100"
            />

            {/* Modern Biometric Face Reticle & Smart HUD */}
            <FaceViewfinder
              isStreaming={isStreaming}
              isProcessing={isProcessing}
              style={viewfinderStyle}
              isIdle={isIdle}
              presenceDetected={presenceDetected}
              touchlessCountdown={touchlessCountdown}
              touchlessTotal={touchlessDelay}
            />
          </>
        ) : (
          <div className="flex flex-col items-center justify-center text-slate-500 space-y-3 p-6 text-center">
            <div className="w-20 h-20 rounded-full bg-slate-950 flex items-center justify-center border border-slate-800 shadow-lg">
              <CameraOff className="w-10 h-10 text-slate-600" />
            </div>
            <div>
              <p className="text-base font-semibold text-slate-300">Camera Kiosk đang tắt</p>
              <p className="text-xs text-slate-500 mt-1 max-w-sm">
                Vui lòng kích hoạt camera để bắt đầu vận hành máy cấp giấy thông minh.
              </p>
            </div>
            <button
              onClick={onStartCamera}
              className="mt-2 px-5 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold flex items-center space-x-2 shadow-lg shadow-cyan-600/30 transition transform active:scale-95"
            >
              <Camera className="w-4 h-4" />
              <span>Bật Camera Kiosk</span>
            </button>
          </div>
        )}

        {/* Camera Hardware Error Notice */}
        {cameraError && (
          <div className="absolute bottom-4 inset-x-4 bg-rose-950/90 border border-rose-800 text-rose-300 px-4 py-2.5 rounded-xl text-xs flex items-center space-x-2 backdrop-blur-md">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{cameraError}</span>
          </div>
        )}

        {/* ========================================================================= */}
        {/* RESULT OVERLAY MODAL (Granted, Cooldown Blocked, Mask/Occluded Alert)     */}
        {/* ========================================================================= */}
        {result && (
          <div className="absolute inset-0 bg-slate-950/85 backdrop-blur-md z-20 p-6 flex flex-col items-center justify-center text-center animate-in zoom-in-95 duration-200">
            {result.granted ? (
              // ----------------- SUCCESS / GRANTED -----------------
              <div className="space-y-4 max-w-md">
                <div className="w-20 h-20 mx-auto rounded-3xl bg-emerald-500/20 border-2 border-emerald-500/60 flex items-center justify-center text-4xl shadow-[0_0_40px_rgba(16,185,129,0.3)] animate-bounce">
                  🧻
                </div>
                <div>
                  <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-emerald-950 border border-emerald-700 text-emerald-300 text-xs font-semibold mb-2">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>CẤP GIẤY VỆ SINH THÀNH CÔNG</span>
                  </div>
                  <h3 className="text-xl sm:text-2xl font-black text-white tracking-tight">
                    {result.is_new_user ? 'Chào Mừng Người Dùng Mới!' : `Xin Chào, ${result.name || 'Quý khách'}!`}
                  </h3>
                  <p className="text-xs sm:text-sm text-emerald-400 mt-2 font-medium">
                    {result.message}
                  </p>
                  <p className="text-[11px] text-slate-400 mt-1">
                    Động cơ đang nhả giấy... Vui lòng đón nhận giấy bên dưới khe máy.
                  </p>
                </div>

                {/* Auto dismiss countdown bar */}
                <div className="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-emerald-500 h-full transition-all duration-1000"
                    style={{ width: `${(autoDismissSeconds / 3) * 100}%` }}
                  />
                </div>
                <p className="text-[11px] text-slate-400">
                  Tự động quay lại màn hình chờ sau {autoDismissSeconds}s...
                </p>
              </div>
            ) : result.status === 'COOLDOWN_BLOCKED' ? (
              // ----------------- COOLDOWN BLOCKED -----------------
              <div className="space-y-3 max-w-md">
                <div className="w-20 h-20 mx-auto rounded-3xl bg-amber-500/20 border-2 border-amber-500/60 flex items-center justify-center text-4xl shadow-[0_0_40px_rgba(245,158,11,0.3)]">
                  ⏱️
                </div>
                <div>
                  <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-amber-950 border border-amber-700 text-amber-300 text-xs font-semibold mb-2">
                    <Clock className="w-3.5 h-3.5" />
                    <span>TỪ CHỐI CẤP GIẤY - ĐANG TRONG THỜI GIAN CHỜ</span>
                  </div>
                  <h3 className="text-xl font-bold text-white tracking-tight">
                    {result.name ? `Chào ${result.name}` : 'Bạn Vừa Nhận Giấy!'}
                  </h3>

                  {/* Big Live Countdown Timer */}
                  <div className="my-2.5 py-2.5 px-6 rounded-2xl bg-amber-950/40 border border-amber-800/60">
                    <span className="text-xs text-amber-300 font-semibold block mb-0.5">
                      Vui lòng đợi thêm:
                    </span>
                    <span className="text-3xl font-extrabold font-mono text-amber-400 tracking-wider">
                      {formatSeconds(countdownRemaining)}
                    </span>
                  </div>

                  <p className="text-xs text-slate-300 leading-relaxed">
                    Hệ thống quy định mỗi người chỉ được lấy giấy 1 lần trong vòng <strong>{cooldownMinutes} phút</strong> để tránh lãng phí và lạm dụng giấy vệ sinh công cộng!
                  </p>
                </div>

                {/* Auto dismiss countdown bar */}
                <div className="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-amber-500 h-full transition-all duration-1000"
                    style={{ width: `${(autoDismissSeconds / 3) * 100}%` }}
                  />
                </div>
                <p className="text-[11px] text-slate-400">
                  Tự động quay lại màn hình chờ sau {autoDismissSeconds}s...
                </p>

                <button
                  onClick={() => setResult(null)}
                  className="px-5 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700 transition active:scale-95"
                >
                  Đã hiểu, quay lại ngay
                </button>
              </div>
            ) : (
              // ----------------- MASK / OCCLUDED / NO FACE / SERVER ERROR -----------------
              <div className="space-y-3 max-w-md">
                <div className="w-20 h-20 mx-auto rounded-3xl bg-rose-500/20 border-2 border-rose-500/60 flex items-center justify-center text-4xl shadow-[0_0_40px_rgba(244,63,94,0.3)] animate-pulse">
                  {result.status === 'MASK_DETECTED'
                    ? '😷'
                    : result.status === 'OCCLUSION_DETECTED'
                    ? '✋'
                    : result.status === 'ERROR'
                    ? '⚠️'
                    : '👤'}
                </div>
                <div>
                  <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-rose-950 border border-rose-700 text-rose-300 text-xs font-semibold mb-2">
                    <AlertTriangle className="w-3.5 h-3.5" />
                    <span>
                      {result.status === 'ERROR'
                        ? 'LỖI KẾT NỐI MÁY CHỦ'
                        : 'YÊU CẦU XÁC THỰC KHUÔN MẶT'}
                    </span>
                  </div>
                  <h3 className="text-lg sm:text-xl font-bold text-white">
                    {result.status === 'MASK_DETECTED'
                      ? 'Vui Lòng Tháo Khẩu Trang!'
                      : result.status === 'OCCLUSION_DETECTED'
                      ? 'Khuôn Mặt Bị Che Khuất!'
                      : result.status === 'ERROR'
                      ? 'Chưa Khởi Động Lại Backend Server!'
                      : 'Không Tìm Thấy Khuôn Mặt!'}
                  </h3>
                  <p className="text-xs sm:text-sm text-rose-300 mt-1 font-medium">
                    {result.status === 'ERROR' && result.message === 'Not Found'
                      ? 'FastAPI Backend chưa nhận diện được đường dẫn /api/v1/dispenser. Vui lòng tắt terminal chạy python scripts/run_server.py và chạy lại với cờ --reload!'
                      : result.message}
                  </p>
                </div>

                {/* Auto dismiss countdown bar */}
                <div className="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-rose-500 h-full transition-all duration-1000"
                    style={{ width: `${(autoDismissSeconds / 3) * 100}%` }}
                  />
                </div>
                <p className="text-[11px] text-slate-400">
                  Tự động quay lại màn hình chờ sau {autoDismissSeconds}s...
                </p>

                <button
                  onClick={() => setResult(null)}
                  className="px-5 py-1.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold shadow-lg shadow-rose-600/30 transition active:scale-95"
                >
                  Thử lại ngay
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ========================================================================= */}
      {/* BIG PRIMARY TOUCH BUTTON: "NHẬN GIẤY VỆ SINH"                             */}
      {/* ========================================================================= */}
      <div className="w-full max-w-md flex flex-col items-center mt-3 z-10">
        <button
          disabled={!isActive || isProcessing}
          onClick={handleDispenseClick}
          className={`w-full py-4 px-8 rounded-2xl font-bold text-base sm:text-lg tracking-wide uppercase flex items-center justify-center space-x-3 transition-all transform shadow-2xl ${
            !isActive || isProcessing
              ? 'bg-slate-800 text-slate-500 border border-slate-700 cursor-not-allowed'
              : touchlessEnabled && touchlessCountdown > 0
              ? 'bg-gradient-to-r from-emerald-500 via-cyan-500 to-blue-600 text-white border-2 border-emerald-400 shadow-emerald-500/30 scale-[1.02] animate-pulse'
              : isIdle
              ? 'bg-gradient-to-r from-slate-800 via-slate-850 to-slate-900 hover:from-cyan-900 hover:to-blue-900 text-slate-300 border border-slate-700 hover:border-cyan-500/50'
              : 'bg-gradient-to-r from-cyan-500 via-blue-600 to-indigo-600 hover:from-cyan-400 hover:via-blue-500 hover:to-indigo-500 text-white border-2 border-cyan-400/40 shadow-cyan-500/30 hover:scale-[1.02] active:scale-95'
          }`}
        >
          {isProcessing ? (
            <>
              <RefreshCw className="w-6 h-6 animate-spin text-white" />
              <span>Đang phân tích khuôn mặt...</span>
            </>
          ) : touchlessEnabled && touchlessCountdown > 0 ? (
            <>
              <Scan className="w-6 h-6 animate-spin text-emerald-200" />
              <div className="flex flex-col items-center">
                <span>TỰ ĐỘNG CẤP GIẤY: {touchlessCountdown.toFixed(1)}s</span>
                <span className="text-[10px] font-normal lowercase tracking-normal opacity-90 text-emerald-100">
                  (nhấp để lấy ngay)
                </span>
              </div>
            </>
          ) : (
            <>
              <span className="text-2xl">🧻</span>
              <span>NHẬN GIẤY VỆ SINH</span>
              <Sparkles className="w-5 h-5 text-yellow-300" />
            </>
          )}
        </button>

        <p className="text-[11px] text-slate-400 mt-2 text-center">
          {touchlessEnabled ? (
            <>
              💡 <strong>Chế độ Không Chạm (Touchless):</strong> Đứng trước camera {touchlessDelay}s máy sẽ tự động cấp giấy, hoặc chạm nút để lấy ngay!
            </>
          ) : (
            <>
              💡 <strong>Hướng dẫn:</strong> Nhìn thẳng vào camera và chạm vào nút trên. Hệ thống sẽ tự động xác thực và nhả giấy!
            </>
          )}
        </p>
      </div>

      {/* Bottom status badges */}
      <div className="w-full flex items-center justify-between text-[11px] text-slate-500 mt-3 pt-2 border-t border-slate-900">
        <div className="flex items-center space-x-2">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          <span>Bảo mật ArcFace 512D</span>
        </div>
        <div className="flex items-center space-x-2">
          <UserPlus className="w-3.5 h-3.5 text-cyan-400" />
          <span>Tự động đăng ký người mới</span>
        </div>
      </div>
    </div>
  );
};

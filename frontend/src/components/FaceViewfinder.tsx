import React from 'react';
import { ScanFace, Sparkles, Timer, Moon, UserCheck } from 'lucide-react';

export type ViewfinderStyle = 'hud' | 'corners' | 'oval';

interface FaceViewfinderProps {
  isStreaming: boolean;
  isProcessing: boolean;
  style?: ViewfinderStyle;
  isIdle?: boolean;
  presenceDetected?: boolean;
  touchlessCountdown?: number | null;
  touchlessTotal?: number;
}

export const FaceViewfinder: React.FC<FaceViewfinderProps> = ({
  isStreaming,
  isProcessing,
  style = 'hud',
  isIdle = false,
  presenceDetected = false,
  touchlessCountdown = null,
  touchlessTotal = 1.5,
}) => {
  if (!isStreaming) return null;

  const isCountingDown = touchlessCountdown !== null && touchlessCountdown > 0;
  const progressPercent = isCountingDown && touchlessTotal > 0
    ? Math.max(0, Math.min(100, ((touchlessTotal - touchlessCountdown) / touchlessTotal) * 100))
    : 0;

  return (
    <div className="absolute inset-0 pointer-events-none flex flex-col items-center justify-center select-none overflow-hidden">
      {/* ===================================================================== */}
      {/* STYLE 1: BIOMETRIC AI HUD (Lớp phủ làm mờ hậu cảnh + Góc ngắm + Oval)   */}
      {/* ===================================================================== */}
      {style === 'hud' && (
        <div className="relative flex items-center justify-center -translate-y-3 sm:-translate-y-4">
          {/* Central Face Target Oval with Cinema Cutout Mask */}
          <div
            className={`relative w-44 h-60 sm:w-52 sm:h-72 rounded-[46%] transition-all duration-500 flex items-center justify-center ${
              isProcessing
                ? 'border-2 border-cyan-300 shadow-[0_0_25px_rgba(6,182,212,0.8),0_0_0_9999px_rgba(11,15,25,0.48)] scale-102'
                : isCountingDown
                ? 'border-2 border-emerald-400 shadow-[0_0_25px_rgba(52,211,153,0.7),0_0_0_9999px_rgba(11,15,25,0.42)] scale-102'
                : isIdle
                ? 'border-2 border-slate-600/60 shadow-[0_0_10px_rgba(100,116,139,0.15),0_0_0_9999px_rgba(11,15,25,0.55)] opacity-65 animate-pulse'
                : 'border-2 border-cyan-400/60 shadow-[0_0_16px_rgba(6,182,212,0.25),0_0_0_9999px_rgba(11,15,25,0.40)]'
            }`}
          >
            {/* Top-Left Tech Corner Bracket */}
            <div
              className={`absolute -top-3 -left-3 w-6 h-6 border-t-2 border-l-2 rounded-tl-xl transition-all duration-300 ${
                isProcessing
                  ? 'border-cyan-300 drop-shadow-[0_0_8px_rgba(6,182,212,0.9)] scale-110'
                  : isCountingDown
                  ? 'border-emerald-400 drop-shadow-[0_0_8px_rgba(52,211,153,0.9)] scale-110'
                  : isIdle
                  ? 'border-slate-500/70'
                  : 'border-cyan-400 drop-shadow-[0_0_4px_rgba(6,182,212,0.6)]'
              }`}
            />

            {/* Top-Right Tech Corner Bracket */}
            <div
              className={`absolute -top-3 -right-3 w-6 h-6 border-t-2 border-r-2 rounded-tr-xl transition-all duration-300 ${
                isProcessing
                  ? 'border-cyan-300 drop-shadow-[0_0_8px_rgba(6,182,212,0.9)] scale-110'
                  : isCountingDown
                  ? 'border-emerald-400 drop-shadow-[0_0_8px_rgba(52,211,153,0.9)] scale-110'
                  : isIdle
                  ? 'border-slate-500/70'
                  : 'border-cyan-400 drop-shadow-[0_0_4px_rgba(6,182,212,0.6)]'
              }`}
            />

            {/* Bottom-Left Tech Corner Bracket */}
            <div
              className={`absolute -bottom-3 -left-3 w-6 h-6 border-b-2 border-l-2 rounded-bl-xl transition-all duration-300 ${
                isProcessing
                  ? 'border-cyan-300 drop-shadow-[0_0_8px_rgba(6,182,212,0.9)] scale-110'
                  : isCountingDown
                  ? 'border-emerald-400 drop-shadow-[0_0_8px_rgba(52,211,153,0.9)] scale-110'
                  : isIdle
                  ? 'border-slate-500/70'
                  : 'border-cyan-400 drop-shadow-[0_0_4px_rgba(6,182,212,0.6)]'
              }`}
            />

            {/* Bottom-Right Tech Corner Bracket */}
            <div
              className={`absolute -bottom-3 -right-3 w-6 h-6 border-b-2 border-r-2 rounded-br-xl transition-all duration-300 ${
                isProcessing
                  ? 'border-cyan-300 drop-shadow-[0_0_8px_rgba(6,182,212,0.9)] scale-110'
                  : isCountingDown
                  ? 'border-emerald-400 drop-shadow-[0_0_8px_rgba(52,211,153,0.9)] scale-110'
                  : isIdle
                  ? 'border-slate-500/70'
                  : 'border-cyan-400 drop-shadow-[0_0_4px_rgba(6,182,212,0.6)]'
              }`}
            />

            {/* Precision Cardinal Alignment Ticks */}
            <div className={`absolute -top-2 left-1/2 -translate-x-1/2 w-4 h-0.5 rounded-full transition-colors ${
              isCountingDown ? 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]' : isIdle ? 'bg-slate-600' : 'bg-cyan-400/80 shadow-[0_0_6px_rgba(6,182,212,0.8)]'
            }`} />
            <div className={`absolute -bottom-2 left-1/2 -translate-x-1/2 w-4 h-0.5 rounded-full transition-colors ${
              isCountingDown ? 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]' : isIdle ? 'bg-slate-600' : 'bg-cyan-400/80 shadow-[0_0_6px_rgba(6,182,212,0.8)]'
            }`} />
            <div className={`absolute top-1/2 -left-2 -translate-y-1/2 w-0.5 h-4 rounded-full transition-colors ${
              isCountingDown ? 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]' : isIdle ? 'bg-slate-600' : 'bg-cyan-400/80 shadow-[0_0_6px_rgba(6,182,212,0.8)]'
            }`} />
            <div className={`absolute top-1/2 -right-2 -translate-y-1/2 w-0.5 h-4 rounded-full transition-colors ${
              isCountingDown ? 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]' : isIdle ? 'bg-slate-600' : 'bg-cyan-400/80 shadow-[0_0_6px_rgba(6,182,212,0.8)]'
            }`} />

            {/* Subtle Eye Alignment Landmarks */}
            <div className="absolute top-[38%] inset-x-8 flex justify-between pointer-events-none opacity-40">
              <div className="w-2 h-2 rounded-full border border-cyan-400 flex items-center justify-center">
                <div className="w-0.5 h-0.5 bg-cyan-400 rounded-full" />
              </div>
              <div className="w-2 h-2 rounded-full border border-cyan-400 flex items-center justify-center">
                <div className="w-0.5 h-0.5 bg-cyan-400 rounded-full" />
              </div>
            </div>

            {/* Laser Scanline inside Oval during processing */}
            {isProcessing && (
              <div className="absolute inset-x-4 h-1 bg-gradient-to-r from-transparent via-cyan-300 to-transparent shadow-[0_0_16px_rgba(6,182,212,1)] animate-scan pointer-events-none rounded-full" />
            )}
          </div>
        </div>
      )}

      {/* ===================================================================== */}
      {/* STYLE 2: CYBER CORNERS (Góc ngắm AI tối giản hiện đại)               */}
      {/* ===================================================================== */}
      {style === 'corners' && (
        <div className="relative flex items-center justify-center -translate-y-3 sm:-translate-y-4">
          <div
            className={`relative w-48 h-64 sm:w-56 sm:h-76 transition-all duration-300 flex items-center justify-center ${
              isProcessing || isCountingDown ? 'scale-105' : isIdle ? 'opacity-60' : ''
            }`}
          >
            {/* 4 Large Cyber Corners */}
            <div className={`absolute top-0 left-0 w-9 h-9 border-t-2 border-l-2 rounded-tl-2xl ${
              isCountingDown ? 'border-emerald-400 drop-shadow-[0_0_8px_rgba(52,211,153,0.8)]' : isIdle ? 'border-slate-500' : 'border-cyan-400 drop-shadow-[0_0_8px_rgba(6,182,212,0.8)]'
            }`} />
            <div className={`absolute top-0 right-0 w-9 h-9 border-t-2 border-r-2 rounded-tr-2xl ${
              isCountingDown ? 'border-emerald-400 drop-shadow-[0_0_8px_rgba(52,211,153,0.8)]' : isIdle ? 'border-slate-500' : 'border-cyan-400 drop-shadow-[0_0_8px_rgba(6,182,212,0.8)]'
            }`} />
            <div className={`absolute bottom-0 left-0 w-9 h-9 border-b-2 border-l-2 rounded-bl-2xl ${
              isCountingDown ? 'border-emerald-400 drop-shadow-[0_0_8px_rgba(52,211,153,0.8)]' : isIdle ? 'border-slate-500' : 'border-cyan-400 drop-shadow-[0_0_8px_rgba(6,182,212,0.8)]'
            }`} />
            <div className={`absolute bottom-0 right-0 w-9 h-9 border-b-2 border-r-2 rounded-br-2xl ${
              isCountingDown ? 'border-emerald-400 drop-shadow-[0_0_8px_rgba(52,211,153,0.8)]' : isIdle ? 'border-slate-500' : 'border-cyan-400 drop-shadow-[0_0_8px_rgba(6,182,212,0.8)]'
            }`} />

            {/* Corner Accent Dots */}
            <div className={`absolute top-2.5 left-2.5 w-1.5 h-1.5 rounded-full ${isCountingDown ? 'bg-emerald-400' : isIdle ? 'bg-slate-600' : 'bg-cyan-400'}`} />
            <div className={`absolute top-2.5 right-2.5 w-1.5 h-1.5 rounded-full ${isCountingDown ? 'bg-emerald-400' : isIdle ? 'bg-slate-600' : 'bg-cyan-400'}`} />
            <div className={`absolute bottom-2.5 left-2.5 w-1.5 h-1.5 rounded-full ${isCountingDown ? 'bg-emerald-400' : isIdle ? 'bg-slate-600' : 'bg-cyan-400'}`} />
            <div className={`absolute bottom-2.5 right-2.5 w-1.5 h-1.5 rounded-full ${isCountingDown ? 'bg-emerald-400' : isIdle ? 'bg-slate-600' : 'bg-cyan-400'}`} />

            {/* Center Reticle Crosshair */}
            <div className="w-6 h-6 border border-cyan-400/30 rounded-full flex items-center justify-center">
              <div className={`w-1 h-1 rounded-full ${isCountingDown ? 'bg-emerald-400' : isIdle ? 'bg-slate-600' : 'bg-cyan-400/60'}`} />
            </div>

            {/* Laser Scanline */}
            {isProcessing && (
              <div className="absolute inset-x-2 h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_16px_rgba(6,182,212,1)] animate-scan rounded-full" />
            )}
          </div>
        </div>
      )}

      {/* ===================================================================== */}
      {/* STYLE 3: SMART OVAL (Vòng sinh trắc học Apple FaceID)                */}
      {/* ===================================================================== */}
      {style === 'oval' && (
        <div className="relative flex items-center justify-center -translate-y-3 sm:-translate-y-4">
          <div
            className={`w-44 h-60 sm:w-52 sm:h-72 rounded-full border-2 transition-all duration-500 flex items-center justify-center ${
              isProcessing
                ? 'border-cyan-300 shadow-[0_0_30px_rgba(6,182,212,0.7)] scale-105'
                : isCountingDown
                ? 'border-emerald-400 shadow-[0_0_30px_rgba(52,211,153,0.7)] scale-105'
                : isIdle
                ? 'border-slate-600/60 opacity-60'
                : 'border-cyan-400/60 shadow-[0_0_15px_rgba(6,182,212,0.25)]'
            }`}
          >
            {/* Concentric Inner Ring */}
            <div className="w-[92%] h-[92%] rounded-full border border-cyan-500/30 border-dashed" />

            {/* Center Focus Dot */}
            <div className={`absolute w-2 h-2 rounded-full animate-ping ${isCountingDown ? 'bg-emerald-400' : 'bg-cyan-400/60'}`} />

            {/* Laser Scanline */}
            {isProcessing && (
              <div className="absolute inset-x-4 h-1 bg-gradient-to-r from-transparent via-cyan-300 to-transparent shadow-[0_0_16px_rgba(6,182,212,1)] animate-scan rounded-full" />
            )}
          </div>
        </div>
      )}

      {/* ===================================================================== */}
      {/* INSTRUCTION PILL (Căn chính giữa tuyệt đối 100%, không bị lệch)       */}
      {/* ===================================================================== */}
      <div className="absolute bottom-3 sm:bottom-4 left-1/2 -translate-x-1/2 z-20 pointer-events-none flex flex-col items-center">
        {/* Touchless Progress Bar when Counting Down */}
        {isCountingDown && (
          <div className="w-48 h-1.5 bg-slate-900/90 rounded-full border border-emerald-500/40 overflow-hidden mb-1.5 shadow-lg">
            <div
              className="h-full bg-gradient-to-r from-emerald-500 to-cyan-400 transition-all duration-100 ease-linear rounded-full"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        )}

        <div className={`flex items-center space-x-2 backdrop-blur-md px-4 py-1.5 rounded-full shadow-2xl text-xs font-medium tracking-wide whitespace-nowrap transition-all duration-300 ${
          isProcessing
            ? 'bg-slate-950/90 border border-cyan-500/50 text-cyan-200 shadow-black/80'
            : isCountingDown
            ? 'bg-emerald-950/90 border border-emerald-500/60 text-emerald-200 shadow-emerald-950/50 scale-102'
            : isIdle
            ? 'bg-slate-950/80 border border-slate-700/60 text-slate-400 shadow-black/60'
            : 'bg-slate-950/90 border border-cyan-500/50 text-cyan-200 shadow-black/80'
        }`}>
          {isProcessing ? (
            <>
              <Sparkles className="w-3.5 h-3.5 text-cyan-300 animate-spin shrink-0" />
              <span className="font-semibold text-cyan-300">Đang quét sinh trắc học AI...</span>
            </>
          ) : isCountingDown ? (
            <>
              <Timer className="w-3.5 h-3.5 text-emerald-400 animate-spin shrink-0" />
              <span className="font-semibold text-emerald-300">
                Tự động cấp giấy sau: {touchlessCountdown.toFixed(1)}s
              </span>
            </>
          ) : isIdle ? (
            <>
              <Moon className="w-3.5 h-3.5 text-slate-400 shrink-0" />
              <span>Xin chào! Bước lại gần để nhận giấy</span>
            </>
          ) : presenceDetected ? (
            <>
              <UserCheck className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
              <span>Đã thấy khuôn mặt • Bấm nút nhận giấy</span>
            </>
          ) : (
            <>
              <ScanFace className="w-3.5 h-3.5 text-cyan-400 animate-pulse shrink-0" />
              <span>Đặt khuôn mặt vào tâm ngắm</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

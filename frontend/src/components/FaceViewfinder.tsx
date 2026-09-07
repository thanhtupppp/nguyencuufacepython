import React from 'react';
import { ScanFace, Sparkles, SlidersHorizontal } from 'lucide-react';

export type ViewfinderStyle = 'hud' | 'corners' | 'oval';

interface FaceViewfinderProps {
  isStreaming: boolean;
  isProcessing: boolean;
  style?: ViewfinderStyle;
  onCycleStyle?: () => void;
}

export const FaceViewfinder: React.FC<FaceViewfinderProps> = ({
  isStreaming,
  isProcessing,
  style = 'hud',
  onCycleStyle,
}) => {
  if (!isStreaming) return null;

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
                : 'border-2 border-cyan-400/50 shadow-[0_0_15px_rgba(6,182,212,0.2),0_0_0_9999px_rgba(11,15,25,0.40)]'
            }`}
          >
            {/* Top-Left Tech Corner Bracket */}
            <div
              className={`absolute -top-3 -left-3 w-6 h-6 border-t-2 border-l-2 rounded-tl-xl transition-all duration-300 ${
                isProcessing ? 'border-cyan-300 drop-shadow-[0_0_8px_rgba(6,182,212,0.9)] scale-110' : 'border-cyan-400 drop-shadow-[0_0_4px_rgba(6,182,212,0.6)]'
              }`}
            />

            {/* Top-Right Tech Corner Bracket */}
            <div
              className={`absolute -top-3 -right-3 w-6 h-6 border-t-2 border-r-2 rounded-tr-xl transition-all duration-300 ${
                isProcessing ? 'border-cyan-300 drop-shadow-[0_0_8px_rgba(6,182,212,0.9)] scale-110' : 'border-cyan-400 drop-shadow-[0_0_4px_rgba(6,182,212,0.6)]'
              }`}
            />

            {/* Bottom-Left Tech Corner Bracket */}
            <div
              className={`absolute -bottom-3 -left-3 w-6 h-6 border-b-2 border-l-2 rounded-bl-xl transition-all duration-300 ${
                isProcessing ? 'border-cyan-300 drop-shadow-[0_0_8px_rgba(6,182,212,0.9)] scale-110' : 'border-cyan-400 drop-shadow-[0_0_4px_rgba(6,182,212,0.6)]'
              }`}
            />

            {/* Bottom-Right Tech Corner Bracket */}
            <div
              className={`absolute -bottom-3 -right-3 w-6 h-6 border-b-2 border-r-2 rounded-br-xl transition-all duration-300 ${
                isProcessing ? 'border-cyan-300 drop-shadow-[0_0_8px_rgba(6,182,212,0.9)] scale-110' : 'border-cyan-400 drop-shadow-[0_0_4px_rgba(6,182,212,0.6)]'
              }`}
            />

            {/* Precision Cardinal Alignment Ticks */}
            <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-4 h-0.5 bg-cyan-400/80 rounded-full shadow-[0_0_6px_rgba(6,182,212,0.8)]" />
            <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 w-4 h-0.5 bg-cyan-400/80 rounded-full shadow-[0_0_6px_rgba(6,182,212,0.8)]" />
            <div className="absolute top-1/2 -left-2 -translate-y-1/2 w-0.5 h-4 bg-cyan-400/80 rounded-full shadow-[0_0_6px_rgba(6,182,212,0.8)]" />
            <div className="absolute top-1/2 -right-2 -translate-y-1/2 w-0.5 h-4 bg-cyan-400/80 rounded-full shadow-[0_0_6px_rgba(6,182,212,0.8)]" />

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
              isProcessing ? 'scale-105' : ''
            }`}
          >
            {/* 4 Large Cyber Corners */}
            <div className="absolute top-0 left-0 w-9 h-9 border-t-2 border-l-2 border-cyan-400 rounded-tl-2xl drop-shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
            <div className="absolute top-0 right-0 w-9 h-9 border-t-2 border-r-2 border-cyan-400 rounded-tr-2xl drop-shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
            <div className="absolute bottom-0 left-0 w-9 h-9 border-b-2 border-l-2 border-cyan-400 rounded-bl-2xl drop-shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
            <div className="absolute bottom-0 right-0 w-9 h-9 border-b-2 border-r-2 border-cyan-400 rounded-br-2xl drop-shadow-[0_0_8px_rgba(6,182,212,0.8)]" />

            {/* Corner Accent Dots */}
            <div className="absolute top-2.5 left-2.5 w-1.5 h-1.5 rounded-full bg-cyan-400" />
            <div className="absolute top-2.5 right-2.5 w-1.5 h-1.5 rounded-full bg-cyan-400" />
            <div className="absolute bottom-2.5 left-2.5 w-1.5 h-1.5 rounded-full bg-cyan-400" />
            <div className="absolute bottom-2.5 right-2.5 w-1.5 h-1.5 rounded-full bg-cyan-400" />

            {/* Center Reticle Crosshair */}
            <div className="w-6 h-6 border border-cyan-400/30 rounded-full flex items-center justify-center">
              <div className="w-1 h-1 bg-cyan-400/60 rounded-full" />
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
                : 'border-cyan-400/60 shadow-[0_0_15px_rgba(6,182,212,0.25)]'
            }`}
          >
            {/* Concentric Inner Ring */}
            <div className="w-[92%] h-[92%] rounded-full border border-cyan-500/30 border-dashed" />

            {/* Center Focus Dot */}
            <div className="absolute w-2 h-2 rounded-full bg-cyan-400/60 animate-ping" />

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
      <div className="absolute bottom-3 sm:bottom-4 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
        <div className="flex items-center space-x-2 bg-slate-950/90 backdrop-blur-md border border-cyan-500/50 text-cyan-200 px-4 py-1.5 rounded-full shadow-2xl shadow-black/80 text-xs font-medium tracking-wide whitespace-nowrap">
          {isProcessing ? (
            <>
              <Sparkles className="w-3.5 h-3.5 text-cyan-300 animate-spin shrink-0" />
              <span className="font-semibold text-cyan-300">Đang quét sinh trắc học AI...</span>
            </>
          ) : (
            <>
              <ScanFace className="w-3.5 h-3.5 text-cyan-400 animate-pulse shrink-0" />
              <span>Đặt khuôn mặt vào tâm ngắm</span>
            </>
          )}
        </div>
      </div>

      {/* ===================================================================== */}
      {/* STYLE SWITCHER BUTTON (Nằm riêng biệt ở góc dưới bên phải)            */}
      {/* ===================================================================== */}
      {onCycleStyle && (
        <div className="absolute bottom-3 sm:bottom-4 right-3 z-20 pointer-events-auto">
          <button
            onClick={onCycleStyle}
            className="px-2.5 py-1.5 bg-slate-950/85 hover:bg-slate-900 border border-slate-700/80 hover:border-cyan-500/50 text-slate-300 hover:text-cyan-300 rounded-full text-[10px] font-mono transition shadow-xl flex items-center space-x-1"
            title="Nhấp để chuyển kiểu khung ngắm (HUD / Góc ngắm / Vòng tròn)"
          >
            <SlidersHorizontal className="w-3 h-3 text-cyan-400 shrink-0" />
            <span className="text-slate-400">HUD:</span>
            <span className="font-bold text-cyan-400 uppercase">{style}</span>
          </button>
        </div>
      )}
    </div>
  );
};

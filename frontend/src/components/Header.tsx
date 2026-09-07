import React from 'react';
import { ShieldCheck, Wifi, WifiOff, Users, UserPlus, Server, Settings } from 'lucide-react';
import { ConnectionStatus } from '../types';

interface HeaderProps {
  apiOnline: boolean;
  wsStatus: ConnectionStatus;
  totalPersons: number;
  onOpenPersons: () => void;
  onOpenEnroll: () => void;
  onOpenSettings: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  apiOnline,
  wsStatus,
  totalPersons,
  onOpenPersons,
  onOpenEnroll,
  onOpenSettings,
}) => {
  return (
    <header className="bg-slate-900/80 backdrop-blur-md border-b border-slate-800 sticky top-0 z-30 px-4 lg:px-8 py-3 transition-colors">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
        {/* Left: Brand / System info */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-500 flex items-center justify-center shadow-lg shadow-cyan-500/20 ring-1 ring-cyan-400/30">
            <ShieldCheck className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-lg font-bold text-white tracking-wide">
                NguyenCuuFace <span className="text-cyan-400 font-semibold text-sm">AI SYSTEM</span>
              </h1>
              <span className="px-1.5 py-0.5 text-[10px] font-mono tracking-wider uppercase bg-cyan-950/80 text-cyan-300 border border-cyan-800/60 rounded">
                v1.2 Live
              </span>
            </div>
            <p className="text-xs text-slate-400">Hệ thống Giám sát & Nhận diện Khuôn mặt Thời gian thực</p>
          </div>
        </div>

        {/* Center/Right: Statuses & Action Buttons */}
        <div className="flex flex-wrap items-center gap-3">
          {/* API Health Pill */}
          <div
            className={`flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-medium border ${
              apiOnline
                ? 'bg-emerald-950/50 border-emerald-800 text-emerald-300'
                : 'bg-rose-950/50 border-rose-800 text-rose-300'
            }`}
            title={`FastAPI Server: ${apiOnline ? 'Online' : 'Offline'}`}
          >
            <Server className="w-3.5 h-3.5" />
            <span>API: {apiOnline ? 'Online' : 'Mất kết nối'}</span>
            <span
              className={`w-2 h-2 rounded-full ${
                apiOnline ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'
              }`}
            />
          </div>

          {/* WebSocket Status Pill */}
          <div
            className={`flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-medium border ${
              wsStatus === 'connected'
                ? 'bg-cyan-950/50 border-cyan-800 text-cyan-300'
                : wsStatus === 'connecting'
                ? 'bg-amber-950/50 border-amber-800 text-amber-300'
                : 'bg-rose-950/50 border-rose-800 text-rose-300'
            }`}
            title={`WebSocket Events: ${wsStatus}`}
          >
            {wsStatus === 'connected' ? (
              <Wifi className="w-3.5 h-3.5 text-cyan-400" />
            ) : (
              <WifiOff className="w-3.5 h-3.5" />
            )}
            <span>
              WS: {wsStatus === 'connected' ? 'Đã kết nối' : wsStatus === 'connecting' ? 'Đang nối...' : 'Ngắt kết nối'}
            </span>
            <span
              className={`w-2 h-2 rounded-full ${
                wsStatus === 'connected'
                  ? 'bg-cyan-400 animate-ping'
                  : wsStatus === 'connecting'
                  ? 'bg-amber-400 animate-pulse'
                  : 'bg-rose-500'
              }`}
            />
          </div>

          {/* Persons Drawer Button */}
          <button
            onClick={onOpenPersons}
            className="flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition shadow-sm"
          >
            <Users className="w-3.5 h-3.5 text-slate-400" />
            <span>Nhân sự</span>
            <span className="px-1.5 py-0.2 bg-slate-900 text-cyan-300 rounded font-mono text-[11px]">
              {totalPersons}
            </span>
          </button>

          {/* Settings Button */}
          <button
            onClick={onOpenSettings}
            className="p-1.5 rounded-lg text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 border border-slate-700 transition"
            title="Cài đặt thông số Kiosk & Chống lạm dụng"
          >
            <Settings className="w-4 h-4 text-cyan-400" />
          </button>

          {/* Enroll Modal Button */}
          <button
            onClick={onOpenEnroll}
            className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg text-xs font-medium bg-cyan-600 hover:bg-cyan-500 text-white shadow-md shadow-cyan-600/30 transition transform active:scale-95"
          >
            <UserPlus className="w-3.5 h-3.5" />
            <span>Đăng ký mới</span>
          </button>
        </div>
      </div>
    </header>
  );
};

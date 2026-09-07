import React, { useState } from 'react';
import { ShieldCheck, UserX, AlertTriangle, EyeOff, Radio, Trash2, Filter } from 'lucide-react';
import { AccessEvent } from '../types';

interface EventStreamProps {
  events: AccessEvent[];
  onClear: () => void;
}

export const EventStream: React.FC<EventStreamProps> = ({ events, onClear }) => {
  const [filter, setFilter] = useState<'all' | 'matched' | 'alerts'>('all');

  const filteredEvents = events.filter((ev) => {
    if (filter === 'matched') return ev.status === 'MATCHED';
    if (filter === 'alerts') return ev.status !== 'MATCHED';
    return true;
  });

  const formatTime = (ts: string) => {
    try {
      const date = new Date(ts);
      return date.toLocaleTimeString('vi-VN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch {
      return ts;
    }
  };

  const renderStatusBadge = (status: string) => {
    switch (status) {
      case 'MATCHED':
        return (
          <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-950/80 border border-emerald-700/60 text-emerald-300 flex items-center space-x-1">
            <ShieldCheck className="w-3 h-3 text-emerald-400" />
            <span>Hợp lệ</span>
          </span>
        );
      case 'UNKNOWN':
        return (
          <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-amber-950/80 border border-amber-700/60 text-amber-300 flex items-center space-x-1">
            <UserX className="w-3 h-3 text-amber-400" />
            <span>Người lạ</span>
          </span>
        );
      case 'MASK_DETECTED':
        return (
          <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-rose-950/80 border border-rose-700/60 text-rose-300 flex items-center space-x-1">
            <AlertTriangle className="w-3 h-3 text-rose-400" />
            <span>Khẩu trang</span>
          </span>
        );
      case 'OCCLUDED_FACE':
        return (
          <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-rose-950/80 border border-rose-700/60 text-rose-300 flex items-center space-x-1">
            <EyeOff className="w-3 h-3 text-rose-400" />
            <span>Mặt bị che</span>
          </span>
        );
      case 'SPOOF_DETECTED':
        return (
          <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-red-950/90 border border-red-600 text-red-300 flex items-center space-x-1">
            <AlertTriangle className="w-3 h-3 text-red-400" />
            <span>Giả mạo!</span>
          </span>
        );
      default:
        return (
          <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-slate-800 text-slate-300">
            {status}
          </span>
        );
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-cyan-950/60 border border-cyan-800/40 text-cyan-400">
            <Radio className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-semibold tracking-wide text-slate-200">
              NHẬT KÝ RA VÀO (EVENT STREAM)
            </h2>
            <p className="text-[11px] text-slate-400">Thời gian thực qua WebSocket Hub</p>
          </div>
        </div>

        {events.length > 0 && (
          <button
            onClick={onClear}
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-500 hover:text-rose-400 transition"
            title="Xóa lịch sử hiển thị"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center space-x-1.5 my-3">
        <Filter className="w-3.5 h-3.5 text-slate-500 mr-1" />
        <button
          onClick={() => setFilter('all')}
          className={`px-2.5 py-1 rounded-lg text-xs font-medium transition ${
            filter === 'all'
              ? 'bg-cyan-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
          }`}
        >
          Tất cả ({events.length})
        </button>
        <button
          onClick={() => setFilter('matched')}
          className={`px-2.5 py-1 rounded-lg text-xs font-medium transition ${
            filter === 'matched'
              ? 'bg-emerald-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
          }`}
        >
          Đã xác thực ({events.filter((e) => e.status === 'MATCHED').length})
        </button>
        <button
          onClick={() => setFilter('alerts')}
          className={`px-2.5 py-1 rounded-lg text-xs font-medium transition ${
            filter === 'alerts'
              ? 'bg-rose-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
          }`}
        >
          Cảnh báo ({events.filter((e) => e.status !== 'MATCHED').length})
        </button>
      </div>

      {/* Event Cards Scroll Area */}
      <div className="flex-1 overflow-y-auto space-y-2.5 pr-1 max-h-[500px]">
        {filteredEvents.length === 0 ? (
          <div className="h-48 flex flex-col items-center justify-center text-slate-500 text-xs">
            <Radio className="w-8 h-8 text-slate-700 mb-2 animate-pulse" />
            <span>Chưa có sự kiện nào được ghi nhận</span>
            <span className="text-[11px] text-slate-600 mt-1">
              Bật camera hoặc phát luồng RTSP để nhận diện
            </span>
          </div>
        ) : (
          filteredEvents.map((ev, index) => {
            const isMatch = ev.status === 'MATCHED';
            return (
              <div
                key={ev.id || index}
                className={`p-3 rounded-xl border transition-all ${
                  isMatch
                    ? 'bg-slate-950/60 border-slate-800/80 hover:border-emerald-700/50'
                    : 'bg-rose-950/20 border-rose-900/30 hover:border-rose-700/60'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  {/* Left Identity Info */}
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-semibold text-white">
                        {ev.name || ev.person_id || 'Chưa xác định'}
                      </span>
                      {ev.person_id && ev.name && (
                        <span className="text-[10px] font-mono text-slate-400 bg-slate-800/80 px-1.5 py-0.5 rounded">
                          {ev.person_id}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center space-x-2 text-[11px] text-slate-400 mt-0.5">
                      <span>Thiết bị: {ev.device_id || 'Camera'}</span>
                      {ev.similarity !== undefined && ev.similarity !== null && (
                        <span>
                          • Độ khớp: {(ev.similarity * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Right Status & Time */}
                  <div className="flex flex-col items-end space-y-1">
                    {renderStatusBadge(ev.status)}
                    <span className="text-[10px] font-mono text-slate-500">
                      {formatTime(ev.timestamp)}
                    </span>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
